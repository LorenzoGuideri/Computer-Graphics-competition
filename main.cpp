#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 texCoords;
  glm::vec4 boneWeights;
  glm::ivec4 boneIDs;
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  GLuint VAO, VBO, EBO;
  int materialIndex;
  glm::mat4 nodeTransform;
};

struct Material {
  GLuint diffuseTexture;
  glm::vec3 diffuseColor;
  glm::vec3 specularColor;
  float shininess;
};

struct KeyFrame {
  float timeStamp;
  glm::vec3 position;
  glm::quat rotation;
  glm::vec3 scale;
};

struct BoneAnimation {
  std::string boneName;
  std::vector<KeyFrame> keyFrames;
};

struct Animation {
  std::string name;
  float duration;
  float ticksPerSecond;
  std::vector<BoneAnimation> boneAnimations;
};

struct SceneData {
  std::vector<Mesh> meshes;
  std::vector<Material> materials;
  std::map<std::string, GLuint> textures;
  std::vector<Animation> animations;
  std::map<std::string, int> boneMapping;
  std::vector<glm::mat4> boneOffsets;
  std::vector<glm::mat4> boneTransforms;
  glm::mat4 globalInverseTransform;
  const aiScene *scene;
};

struct VideoEncoder {
  AVFormatContext *formatContext;
  AVCodecContext *codecContext;
  AVStream *stream;
  AVFrame *frame;
  SwsContext *swsContext;
  uint8_t *frameBuffer;
  int frameCount;
};

const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 aBoneWeights;
layout (location = 4) in ivec4 aBoneIDs;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 boneTransforms[1000];

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

void main() {
    vec4 totalPosition = vec4(0.0);
    vec3 totalNormal = vec3(0.0);

    float totalWeight = aBoneWeights.x + aBoneWeights.y + aBoneWeights.z + aBoneWeights.w;
    if (totalWeight > 0.0) {
        for (int i = 0; i < 4; i++) {
            if (aBoneWeights[i] > 0.0) {
                vec4 localPos = boneTransforms[aBoneIDs[i]] * vec4(aPos, 1.0);
                totalPosition += localPos * aBoneWeights[i];
                mat3 normalMatrix = mat3(boneTransforms[aBoneIDs[i]]);
                totalNormal += normalMatrix * aNormal * aBoneWeights[i];
            }
        }
    } else {
        totalPosition = vec4(aPos, 1.0);
        totalNormal = aNormal;
    }

    gl_Position = projection * view * model * totalPosition;
    FragPos = vec3(model * totalPosition);
    Normal = mat3(transpose(inverse(model))) * totalNormal;
    TexCoords = aTexCoords;
}
)";

const char *fragmentShaderSource = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform sampler2D diffuseTexture;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 diffuseColor;

out vec4 FragColor;

void main() {
    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec4 texColor = texture(diffuseTexture, TexCoords);

    vec3 result = (ambient + diffuse) * texColor.rgb;
    FragColor = vec4(result, texColor.a);
}
)";

glm::mat4 convertMatrix(const aiMatrix4x4 &m);
GLuint createShaderProgram();
GLuint loadEmbeddedTexture(const aiTexture *texture);
GLuint loadExternalTexture(const char *filename);
GLuint createDefaultTexture();
void processMaterial(aiMaterial *material, const aiScene *scene,
                     Material &outMaterial);
void processNode(aiNode *node, const aiScene *scene, SceneData &sceneData,
                 glm::mat4 parentTransform = glm::mat4(1.0f));
Mesh processMesh(aiMesh *mesh, const aiScene *scene, SceneData &sceneData);
void processAnimations(const aiScene *scene, SceneData &sceneData);
glm::mat4 interpolateKeyFrames(const std::vector<KeyFrame> &keyFrames,
                               float animationTime);

const BoneAnimation *findBoneAnimation(const Animation &animation,
                                       const std::string &boneName) {
  for (const auto &boneAnim : animation.boneAnimations) {
    if (boneAnim.boneName == boneName)
      return &boneAnim;
  }
  return nullptr;
}

void decomposeTransform(const glm::mat4 &transform, glm::vec3 &position,
                        glm::quat &rotation, glm::vec3 &scale) {
  position = glm::vec3(transform[3]);
  scale.x = glm::length(glm::vec3(transform[0]));
  scale.y = glm::length(glm::vec3(transform[1]));
  scale.z = glm::length(glm::vec3(transform[2]));

  glm::mat3 rotMat(glm::vec3(transform[0]) / scale.x,
                   glm::vec3(transform[1]) / scale.y,
                   glm::vec3(transform[2]) / scale.z);
  rotation = glm::quat_cast(rotMat);
}

glm::mat4 composeTransform(const glm::vec3 &position, const glm::quat &rotation,
                           const glm::vec3 &scale) {
  glm::mat4 transform(1.0f);
  transform = glm::translate(transform, position);
  transform *= glm::mat4_cast(rotation);
  transform = glm::scale(transform, scale);
  return transform;
}

void readNodeHierarchy(float animationTime, const aiNode *node,
                       const glm::mat4 &parentTransform,
                       const Animation &animation, SceneData &sceneData) {
  glm::mat4 nodeTransform = convertMatrix(node->mTransformation);
  const BoneAnimation *boneAnim =
      findBoneAnimation(animation, node->mName.data);
  if (boneAnim) {
    glm::mat4 animatedTransform =
        interpolateKeyFrames(boneAnim->keyFrames, animationTime);
    nodeTransform = animatedTransform;
  }

  glm::mat4 globalTransform = parentTransform * nodeTransform;
  auto it = sceneData.boneMapping.find(node->mName.data);
  if (it != sceneData.boneMapping.end()) {
    int boneIndex = it->second;
    sceneData.boneTransforms[boneIndex] = sceneData.globalInverseTransform *
                                          globalTransform *
                                          sceneData.boneOffsets[boneIndex];
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    readNodeHierarchy(animationTime, node->mChildren[i], globalTransform,
                      animation, sceneData);
  }
}

void updateBoneTransforms(SceneData &sceneData, float timeInSeconds) {
  if (sceneData.animations.empty())
    return;
  const Animation &currentAnim = sceneData.animations[0];

  const float LOOP = 3.0f;
  float ticksPerSecond = currentAnim.ticksPerSecond;

  float timeInTicks = timeInSeconds * ticksPerSecond;
  float loopTicks = LOOP * ticksPerSecond;
  float animationTime = fmod(timeInTicks, loopTicks);

  // Prepare and update bone transforms
  sceneData.boneTransforms.resize(sceneData.boneOffsets.size(),
                                  glm::mat4(1.0f));
  readNodeHierarchy(animationTime, sceneData.scene->mRootNode, glm::mat4(1.0f),
                    currentAnim, sceneData);
}
VideoEncoder initVideoEncoder(const char *filename, int width, int height,
                              int fps) {
  VideoEncoder encoder = {};

  avformat_alloc_output_context2(&encoder.formatContext, NULL, NULL, filename);
  if (!encoder.formatContext) {
    fprintf(stderr, "Could not create output context\n");
    return encoder;
  }

  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
  if (!codec) {
    fprintf(stderr, "Could not find H.264 encoder\n");
    return encoder;
  }

  encoder.stream = avformat_new_stream(encoder.formatContext, codec);
  if (!encoder.stream) {
    fprintf(stderr, "Could not create stream\n");
    return encoder;
  }

  encoder.codecContext = avcodec_alloc_context3(codec);
  encoder.codecContext->width = width;
  encoder.codecContext->height = height;
  encoder.codecContext->time_base = (AVRational){1, fps};
  encoder.codecContext->framerate = (AVRational){fps, 1};
  encoder.codecContext->pix_fmt = AV_PIX_FMT_YUV420P;

  encoder.codecContext->bit_rate = 32000000;       // Base bitrate
  encoder.codecContext->rc_min_rate = 16000000;    // Minimum bitrate
  encoder.codecContext->rc_max_rate = 64000000;    // Maximum bitrate
  encoder.codecContext->rc_buffer_size = 32000000; // Buffer size
  encoder.codecContext->gop_size = 12;

  if (avcodec_open2(encoder.codecContext, codec, NULL) < 0) {
    fprintf(stderr, "Could not open codec\n");
    return encoder;
  }

  if (avcodec_parameters_from_context(encoder.stream->codecpar,
                                      encoder.codecContext) < 0) {
    fprintf(stderr, "Failed to copy codec parameters to stream\n");
    return encoder;
  }

  encoder.frameBuffer = (uint8_t *)malloc(width * height * 3);
  encoder.frame = av_frame_alloc();
  encoder.frame->format = encoder.codecContext->pix_fmt;
  encoder.frame->width = width;
  encoder.frame->height = height;
  av_frame_get_buffer(encoder.frame, 0);

  encoder.swsContext =
      sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height,
                     AV_PIX_FMT_YUV420P, SWS_BILINEAR, NULL, NULL, NULL);

  return encoder;
}

void captureAndEncodeFrame(GLFWwindow *window, VideoEncoder &encoder) {
  glReadPixels(0, 0, encoder.codecContext->width, encoder.codecContext->height,
               GL_RGB, GL_UNSIGNED_BYTE, encoder.frameBuffer);

  const uint8_t *const rgbData[1] = {encoder.frameBuffer};
  const int rgbStride[1] = {3 * encoder.codecContext->width};
  sws_scale(encoder.swsContext, rgbData, rgbStride, 0,
            encoder.codecContext->height, encoder.frame->data,
            encoder.frame->linesize);

  encoder.frame->pts = encoder.frameCount++;

  AVPacket *packet = av_packet_alloc();
  int ret = avcodec_send_frame(encoder.codecContext, encoder.frame);
  if (ret < 0) {
    fprintf(stderr, "Error sending frame for encoding\n");
    av_packet_free(&packet);
    return;
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(encoder.codecContext, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      break;
    if (ret < 0) {
      fprintf(stderr, "Error receiving packet from encoder\n");
      break;
    }

    av_packet_rescale_ts(packet, encoder.codecContext->time_base,
                         encoder.stream->time_base);
    av_interleaved_write_frame(encoder.formatContext, packet);
  }

  av_packet_free(&packet);
}

glm::mat4 convertMatrix(const aiMatrix4x4 &m) {
  return glm::mat4(m.a1, m.b1, m.c1, m.d1, m.a2, m.b2, m.c2, m.d2, m.a3, m.b3,
                   m.c3, m.d3, m.a4, m.b4, m.c4, m.d4);
}

GLuint createShaderProgram() {
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    fprintf(stderr, "Vertex shader compilation failed: %s\n", infoLog);
    return 0;
  }

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    fprintf(stderr, "Fragment shader compilation failed: %s\n", infoLog);
    return 0;
  }

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    fprintf(stderr, "Shader program linking failed: %s\n", infoLog);
    return 0;
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
  return shaderProgram;
}

GLuint createDefaultTexture() {
  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);

  const int width = 64, height = 64;
  unsigned char checkerboard[width][height][4];
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      unsigned char c = ((i & 8) == (j & 8)) ? 255 : 128;
      checkerboard[i][j][0] = c;
      checkerboard[i][j][1] = c;
      checkerboard[i][j][2] = c;
      checkerboard[i][j][3] = 255;
    }
  }

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, checkerboard);
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  return textureID;
}

GLuint loadEmbeddedTexture(const aiTexture *texture) {
  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  if (texture->mHeight == 0) {
    int width, height, channels;
    unsigned char *data = stbi_load_from_memory(
        reinterpret_cast<const unsigned char *>(texture->pcData),
        texture->mWidth, &width, &height, &channels, 4);

    if (data) {
      GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
      glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                   GL_UNSIGNED_BYTE, data);
      glGenerateMipmap(GL_TEXTURE_2D);
      stbi_image_free(data);
    } else {
      return createDefaultTexture();
    }
  } else {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture->mWidth, texture->mHeight, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, texture->pcData);
    glGenerateMipmap(GL_TEXTURE_2D);
  }

  return textureID;
}

GLuint loadExternalTexture(const char *filename) {
  GLuint textureID;
  glGenTextures(1, &textureID);

  int width, height, channels;
  // Force load 4 channels for consistency
  unsigned char *data = stbi_load(filename, &width, &height, &channels, 4);
  if (data) {
    GLenum format = GL_RGBA; // Since we forced 4 channels, use GL_RGBA
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                 GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
  } else {
    fprintf(stderr, "Failed to load texture: %s\n", filename);
    return createDefaultTexture();
  }

  return textureID;
}

void processMaterial(aiMaterial *material, const aiScene *scene,
                     Material &outMaterial) {
  aiColor3D color;
  float shininess;

  if (material->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
    outMaterial.diffuseColor = glm::vec3(color.r, color.g, color.b);
  } else {
    outMaterial.diffuseColor = glm::vec3(0.8f);
  }

  if (material->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
    outMaterial.specularColor = glm::vec3(color.r, color.g, color.b);
  } else {
    outMaterial.specularColor = glm::vec3(0.2f);
  }

  if (material->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
    outMaterial.shininess = shininess;
  } else {
    outMaterial.shininess = 32.0f;
  }

  // Try base color first
  if (material->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
    aiString texturePath;
    if (material->GetTexture(aiTextureType_BASE_COLOR, 0, &texturePath) ==
        AI_SUCCESS) {
      const aiTexture *embeddedTex =
          scene->GetEmbeddedTexture(texturePath.C_Str());
      if (embeddedTex)
        outMaterial.diffuseTexture = loadEmbeddedTexture(embeddedTex);
      else
        outMaterial.diffuseTexture = loadExternalTexture(texturePath.C_Str());
      return;
    }
  }

  // Fallback to diffuse
  if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
    aiString texPath;
    if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) ==
        AI_SUCCESS) {
      const aiTexture *embeddedTex = scene->GetEmbeddedTexture(texPath.C_Str());
      if (embeddedTex)
        outMaterial.diffuseTexture = loadEmbeddedTexture(embeddedTex);
      else
        outMaterial.diffuseTexture = loadExternalTexture(texPath.C_Str());
      return;
    }
  }

  // No texture found
  outMaterial.diffuseTexture = createDefaultTexture();
}

void processNode(aiNode *node, const aiScene *scene, SceneData &sceneData,
                 glm::mat4 parentTransform) {
  glm::mat4 nodeTransform = convertMatrix(node->mTransformation);
  glm::mat4 globalTransform = parentTransform * nodeTransform;

  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    Mesh newMesh = processMesh(mesh, scene, sceneData);
    newMesh.nodeTransform = globalTransform;
    sceneData.meshes.push_back(newMesh);
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    processNode(node->mChildren[i], scene, sceneData, globalTransform);
  }
}

Mesh processMesh(aiMesh *mesh, const aiScene *scene, SceneData &sceneData) {
  Mesh newMesh;
  newMesh.vertices.resize(mesh->mNumVertices);

  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    Vertex &vertex = newMesh.vertices[i];
    vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y,
                                mesh->mVertices[i].z);
    vertex.normal = mesh->HasNormals()
                        ? glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y,
                                    mesh->mNormals[i].z)
                        : glm::vec3(0.0f);
    if (mesh->mTextureCoords[0]) {
      vertex.texCoords =
          glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
    } else {
      vertex.texCoords = glm::vec2(0.0f);
    }

    vertex.boneWeights = glm::vec4(0.0f);
    vertex.boneIDs = glm::ivec4(-1);
  }

  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (unsigned int j = 0; j < face.mNumIndices; j++) {
      newMesh.indices.push_back(face.mIndices[j]);
    }
  }

  for (unsigned int i = 0; i < mesh->mNumBones; i++) {
    aiBone *bone = mesh->mBones[i];
    std::string boneName(bone->mName.data);
    int boneIndex;
    if (sceneData.boneMapping.find(boneName) == sceneData.boneMapping.end()) {
      boneIndex = (int)sceneData.boneMapping.size();
      sceneData.boneMapping[boneName] = boneIndex;
      sceneData.boneOffsets.push_back(convertMatrix(bone->mOffsetMatrix));
    } else {
      boneIndex = sceneData.boneMapping[boneName];
    }

    for (unsigned int j = 0; j < bone->mNumWeights; j++) {
      unsigned int vertexID = bone->mWeights[j].mVertexId;
      float weight = bone->mWeights[j].mWeight;
      for (int k = 0; k < 4; k++) {
        if (newMesh.vertices[vertexID].boneWeights[k] == 0.0f) {
          newMesh.vertices[vertexID].boneWeights[k] = weight;
          newMesh.vertices[vertexID].boneIDs[k] = boneIndex;
          break;
        }
      }
    }
  }

  for (auto &v : newMesh.vertices) {
    float totalWeight =
        v.boneWeights.x + v.boneWeights.y + v.boneWeights.z + v.boneWeights.w;
    if (totalWeight > 0.0f) {
      v.boneWeights /= totalWeight;
    }
  }

  glGenVertexArrays(1, &newMesh.VAO);
  glGenBuffers(1, &newMesh.VBO);
  glGenBuffers(1, &newMesh.EBO);

  glBindVertexArray(newMesh.VAO);
  glBindBuffer(GL_ARRAY_BUFFER, newMesh.VBO);
  glBufferData(GL_ARRAY_BUFFER, newMesh.vertices.size() * sizeof(Vertex),
               &newMesh.vertices[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, newMesh.EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               newMesh.indices.size() * sizeof(unsigned int),
               &newMesh.indices[0], GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)offsetof(Vertex, position));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)offsetof(Vertex, normal));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)offsetof(Vertex, texCoords));
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)offsetof(Vertex, boneWeights));
  glEnableVertexAttribArray(4);
  glVertexAttribIPointer(4, 4, GL_INT, sizeof(Vertex),
                         (void *)offsetof(Vertex, boneIDs));
  glBindVertexArray(0);

  newMesh.materialIndex = mesh->mMaterialIndex;
  return newMesh;
}

void processAnimations(const aiScene *scene, SceneData &sceneData) {
  for (unsigned int i = 0; i < scene->mNumAnimations; i++) {
    aiAnimation *aiAnim = scene->mAnimations[i];
    Animation animation;
    animation.name = aiAnim->mName.data;
    animation.duration = (float)aiAnim->mDuration;
    animation.ticksPerSecond =
        aiAnim->mTicksPerSecond != 0 ? (float)aiAnim->mTicksPerSecond : 25.0f;

    for (unsigned int j = 0; j < aiAnim->mNumChannels; j++) {
      aiNodeAnim *channel = aiAnim->mChannels[j];
      BoneAnimation boneAnim;
      boneAnim.boneName = channel->mNodeName.data;

      for (unsigned int k = 0; k < channel->mNumPositionKeys; k++) {
        KeyFrame keyFrame;
        keyFrame.timeStamp = (float)channel->mPositionKeys[k].mTime;

        aiVector3D pos = channel->mPositionKeys[k].mValue;
        keyFrame.position = glm::vec3(pos.x, pos.y, pos.z);

        aiQuaternion rot = channel->mRotationKeys[k].mValue;
        keyFrame.rotation = glm::quat(rot.w, rot.x, rot.y, rot.z);

        aiVector3D scl = channel->mScalingKeys[k].mValue;
        keyFrame.scale = glm::vec3(scl.x, scl.y, scl.z);

        boneAnim.keyFrames.push_back(keyFrame);
      }

      animation.boneAnimations.push_back(boneAnim);
    }

    sceneData.animations.push_back(animation);
  }
}

glm::mat4 interpolateKeyFrames(const std::vector<KeyFrame> &keyFrames,
                               float animationTime) {
  if (keyFrames.empty())
    return glm::mat4(1.0f);

  size_t frameIndex = 0;
  for (size_t i = 0; i < keyFrames.size() - 1; i++) {
    if (animationTime < keyFrames[i + 1].timeStamp) {
      frameIndex = i;
      break;
    }
  }

  float factor =
      (animationTime - keyFrames[frameIndex].timeStamp) /
      (keyFrames[frameIndex + 1].timeStamp - keyFrames[frameIndex].timeStamp);

  glm::vec3 position = glm::mix(keyFrames[frameIndex].position,
                                keyFrames[frameIndex + 1].position, factor);
  glm::quat rotation = glm::slerp(keyFrames[frameIndex].rotation,
                                  keyFrames[frameIndex + 1].rotation, factor);
  glm::vec3 scale = glm::mix(keyFrames[frameIndex].scale,
                             keyFrames[frameIndex + 1].scale, factor);

  glm::mat4 transform(1.0f);
  transform = glm::translate(transform, position);
  transform *= glm::mat4_cast(rotation);
  transform = glm::scale(transform, scale);
  return transform;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr,
            "Usage: %s <bird_model> <scenery_model> <duration_seconds> <fps>\n",
            argv[0]);
    return 1;
  }

  const char *birdModelFile = argv[1];
  const char *sceneryModelFile = argv[2];
  float duration = (float)atof(argv[3]);
  int fps = atoi(argv[4]);

  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  glfwWindowHint(GLFW_RED_BITS, 8);
  glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8);
  glfwWindowHint(GLFW_ALPHA_BITS, 8);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);

  int width = 1920 * 2;
  int height = 1080 * 2;
  GLFWwindow *window = glfwCreateWindow(width, height, "Renderer", NULL, NULL);

  if (!window) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }
  int fbWidth, fbHeight;
  glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
  if (fbWidth != width || fbHeight != height) {
    fprintf(stderr, "Framebuffer size does not match window size\n");
    glfwTerminate();
    return 1;
  }
  glViewport(0, 0, width, height);
  glfwMakeContextCurrent(window);

  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return 1;
  }

  GLuint shaderProgram = createShaderProgram();
  if (!shaderProgram)
    return 1;
  glUseProgram(shaderProgram);

  const aiScene *birdScene =
      aiImportFile(birdModelFile,
                   aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                       aiProcess_CalcTangentSpace | aiProcess_LimitBoneWeights |
                       aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices |
                       aiProcess_ImproveCacheLocality | aiProcess_SortByPType);

  if (!birdScene) {
    fprintf(stderr, "Failed to load bird model: %s\n", aiGetErrorString());
    return 1;
  }

  SceneData sceneDataBird;
  sceneDataBird.scene = birdScene;
  sceneDataBird.globalInverseTransform =
      glm::inverse(convertMatrix(birdScene->mRootNode->mTransformation));
  processNode(birdScene->mRootNode, birdScene, sceneDataBird);
  processAnimations(birdScene, sceneDataBird);
  for (unsigned int i = 0; i < birdScene->mNumMaterials; i++) {
    aiMaterial *aiMat = birdScene->mMaterials[i];
    Material material;
    processMaterial(aiMat, birdScene, material);
    sceneDataBird.materials.push_back(material);
  }

  const aiScene *sceneryScene =
      aiImportFile(sceneryModelFile,
                   aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                       aiProcess_CalcTangentSpace | aiProcess_LimitBoneWeights |
                       aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices |
                       aiProcess_ImproveCacheLocality | aiProcess_SortByPType);

  if (!sceneryScene) {
    fprintf(stderr, "Failed to load scenery model: %s\n", aiGetErrorString());
    return 1;
  }

  SceneData sceneDataScenery;
  sceneDataScenery.scene = sceneryScene;
  sceneDataScenery.globalInverseTransform =
      glm::inverse(convertMatrix(sceneryScene->mRootNode->mTransformation));
  processNode(sceneryScene->mRootNode, sceneryScene, sceneDataScenery);
  processAnimations(sceneryScene, sceneDataScenery);
  for (unsigned int i = 0; i < sceneryScene->mNumMaterials; i++) {
    aiMaterial *aiMat = sceneryScene->mMaterials[i];
    Material material;
    processMaterial(aiMat, sceneryScene, material);
    sceneDataScenery.materials.push_back(material);
  }

  const char *outputFile = "output.mp4";
  VideoEncoder encoder = initVideoEncoder(outputFile, width, height, fps);
  if (!encoder.formatContext) {
    aiReleaseImport(birdScene);
    aiReleaseImport(sceneryScene);
    return 1;
  }

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glViewport(0, 0, width, height);

  // Camera above the bird
  glm::vec3 cameraPos(0.0f, 4.0f, 5.0f);
  glm::vec3 cameraTarget(0.0f, 2.0f, 3.0f);
  glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

  glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                          (float)width / height, 0.1f, 100.0f);
  glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);

  glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1,
                     GL_FALSE, glm::value_ptr(projection));
  glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE,
                     glm::value_ptr(view));

  glm::vec3 lightPos(2.0f, -4.0f, 2.0f);
  glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
  glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1,
               glm::value_ptr(lightPos));
  glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1,
               glm::value_ptr(lightColor));

  if (avio_open(&encoder.formatContext->pb, outputFile, AVIO_FLAG_WRITE) < 0) {
    fprintf(stderr, "Could not open output file\n");
    return 1;
  }
  if (avformat_write_header(encoder.formatContext, NULL) < 0) {
    fprintf(stderr, "Error writing video header\n");
    return 1;
  }

  int totalFrames = (int)(duration * fps);

  // Bird: smaller and rotated, and move it up closer to camera
  glm::mat4 birdModelMatrix = glm::mat4(1.0f);
  birdModelMatrix = glm::scale(birdModelMatrix, glm::vec3(0.002f));
  birdModelMatrix = glm::rotate(birdModelMatrix, glm::radians(90.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
  birdModelMatrix = glm::rotate(birdModelMatrix, glm::radians(30.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f));

  birdModelMatrix =
      glm::translate(birdModelMatrix, glm::vec3(0.0f, 2.0f, 0.0f));

  // Terrain: larger scale
  glm::mat4 sceneryModelMatrix = glm::mat4(1.0f);
  sceneryModelMatrix = glm::rotate(sceneryModelMatrix, glm::radians(90.0f),
                                   glm::vec3(1.0f, 0.0f, 0.0f));
  sceneryModelMatrix = glm::rotate(sceneryModelMatrix, glm::radians(73.5f),
                                   glm::vec3(0.0f, 1.0f, 0.0f));
  sceneryModelMatrix = glm::scale(sceneryModelMatrix, glm::vec3(0.4f));
  sceneryModelMatrix =
      glm::translate(sceneryModelMatrix, glm::vec3(0.0f, -5.0f, 1.5f));

  printf("Starting render of %d frames...\n", totalFrames);

  for (int frame = 0; frame < totalFrames; frame++) {
    float timeInSeconds = frame / (float)fps;

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    updateBoneTransforms(sceneDataBird, timeInSeconds);

    // Move scenery back along -Z to simulate bird forward movement
    glm::mat4 currentSceneryMatrix = glm::translate(
        sceneryModelMatrix,
        glm::vec3(-timeInSeconds * 2.0f, 0.0f, timeInSeconds * 0.55f));

    for (size_t i = 0; i < sceneDataBird.boneTransforms.size(); i++) {
      std::string uniformName = "boneTransforms[" + std::to_string(i) + "]";
      glUniformMatrix4fv(
          glGetUniformLocation(shaderProgram, uniformName.c_str()), 1, GL_FALSE,
          glm::value_ptr(sceneDataBird.boneTransforms[i]));
    }

    // Render bird
    for (const Mesh &mesh : sceneDataBird.meshes) {
      if (mesh.materialIndex < (int)sceneDataBird.materials.size()) {
        const Material &material = sceneDataBird.materials[mesh.materialIndex];
        glBindTexture(GL_TEXTURE_2D, material.diffuseTexture);
        glUniform3fv(glGetUniformLocation(shaderProgram, "diffuseColor"), 1,
                     glm::value_ptr(material.diffuseColor));
        glUniform3fv(glGetUniformLocation(shaderProgram, "specularColor"), 1,
                     glm::value_ptr(material.specularColor));
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"),
                    material.shininess);
      }

      glm::mat4 finalModelMatrix = birdModelMatrix * mesh.nodeTransform;
      glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1,
                         GL_FALSE, glm::value_ptr(finalModelMatrix));

      glBindVertexArray(mesh.VAO);
      glDrawElements(GL_TRIANGLES, (GLsizei)mesh.indices.size(),
                     GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
    }

    // Scenery has no bone animation
    for (size_t i = 0; i < sceneDataScenery.boneTransforms.size(); i++) {
      std::string uniformName = "boneTransforms[" + std::to_string(i) + "]";
      glm::mat4 identity = glm::mat4(1.0f);
      glUniformMatrix4fv(
          glGetUniformLocation(shaderProgram, uniformName.c_str()), 1, GL_FALSE,
          glm::value_ptr(identity));
    }

    // Render scenery
    for (const Mesh &mesh : sceneDataScenery.meshes) {
      if (mesh.materialIndex < (int)sceneDataScenery.materials.size()) {
        const Material &material =
            sceneDataScenery.materials[mesh.materialIndex];
        glBindTexture(GL_TEXTURE_2D, material.diffuseTexture);
        glUniform3fv(glGetUniformLocation(shaderProgram, "diffuseColor"), 1,
                     glm::value_ptr(material.diffuseColor));
        glUniform3fv(glGetUniformLocation(shaderProgram, "specularColor"), 1,
                     glm::value_ptr(material.specularColor));
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"),
                    material.shininess);
      }

      glm::mat4 finalModelMatrix = currentSceneryMatrix * mesh.nodeTransform;
      glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1,
                         GL_FALSE, glm::value_ptr(finalModelMatrix));

      glBindVertexArray(mesh.VAO);
      glDrawElements(GL_TRIANGLES, (GLsizei)mesh.indices.size(),
                     GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
    }

    captureAndEncodeFrame(window, encoder);

    if (frame % 10 == 0) {
      printf("\rRendering progress: %.1f%%",
             (frame + 1) * 100.0f / totalFrames);
      fflush(stdout);
    }
  }

  printf("\nRendering complete!\n");

  av_write_trailer(encoder.formatContext);
  avio_closep(&encoder.formatContext->pb);

  sws_freeContext(encoder.swsContext);
  av_frame_free(&encoder.frame);
  avcodec_free_context(&encoder.codecContext);
  avformat_free_context(encoder.formatContext);
  free(encoder.frameBuffer);

  for (const Mesh &mesh : sceneDataBird.meshes) {
    glDeleteVertexArrays(1, &mesh.VAO);
    glDeleteBuffers(1, &mesh.VBO);
    glDeleteBuffers(1, &mesh.EBO);
  }

  for (const Mesh &mesh : sceneDataScenery.meshes) {
    glDeleteVertexArrays(1, &mesh.VAO);
    glDeleteBuffers(1, &mesh.VBO);
    glDeleteBuffers(1, &mesh.EBO);
  }

  glDeleteProgram(shaderProgram);
  aiReleaseImport(birdScene);
  aiReleaseImport(sceneryScene);
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
