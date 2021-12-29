#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <utility>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

using triplet = std::array<std::uint32_t, 3>;

struct TripletHasher {
    std::size_t operator()(const triplet & a) const {
        std::size_t h = 0;

        for (auto e : a) {
            h ^= std::hash<std::uint32_t>{}(e)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

std::string to_string(std::string_view str)
{
	return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
	throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
	throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

out vec3 position;
out vec3 raw_normal;
out vec2 uv;
out vec3 camera_position;
out mat4 model_;

void main()
{
	gl_Position = projection * view * model * vec4(in_position, 1.0);
	position = (model * vec4(in_position, 1.0)).xyz;
	raw_normal = normalize((model * vec4(in_normal, 0.0)).xyz);
    uv = vec2(1, -1) * in_uv;
    camera_position = (inverse(view) * vec4(0, 0, 0, 1)).xyz;
    model_ = model;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 ambient_color;
uniform vec3 albedo;
uniform vec3 diffuse_color;

uniform vec3 light_direction;
uniform vec3 light_color;

uniform bool has_specular_map;
uniform bool has_diffuse_map;
uniform bool has_normal_map;

uniform mat4 transform;

uniform sampler2D shadow_map;
uniform sampler2D tex;
uniform sampler2D specular_map;
uniform sampler2D diffuse_map;
uniform sampler2D normal_map;

uniform vec3 light_position[3];
uniform vec3 light_colors[3];
uniform vec3 light_attenuation[3];

in vec3 position;
in vec3 raw_normal;
in vec2 uv;
in vec3 camera_position;
in mat4 model_;

layout (location = 0) out vec4 out_color;

vec3 single_lighter(vec3 light_pos, vec3 light_color, vec3 light_att, vec3 normal){
    vec3 dir = light_pos - position;
    vec3 n_dir = normalize(dir);
    float dist = length(dir);

    float cosine = dot(normal, n_dir);
    float intensity = 1.0 / dot(light_att, vec3(1.0, dist, dist * dist));

    vec3 cam_dir = normalize(camera_position - position);

    vec3 reflected = 2.0 * normal * cosine - n_dir;
    vec4 roughness = vec4(0.0);
    if (has_specular_map) {
        roughness = texture(specular_map, uv);
    }
    float specular = pow(max(0.0, dot(reflected, cam_dir)), 24.0) * (1.0 - roughness[0]);

    return light_color * (max(0.0, cosine) * intensity + specular);
}

float get_shadow_factor(vec3 position) {
    vec4 shadow_pos = transform * vec4(position, 1.0);
	shadow_pos /= shadow_pos.w;
	shadow_pos = shadow_pos * 0.5 + vec4(0.5);

    float bias = 0.005;

    vec2 sum = vec2(0.0);
    float sum_w = 0.0;
    const int N = 7;
    float radius = 5.0;
    for (int x = -N; x <= N; ++x) {
        for (int y = -N; y <= N; ++y) {
            float c = exp(-float(x*x + y*y) / (radius*radius));
            sum += c * texture(shadow_map, shadow_pos.xy + vec2(x,y) / vec2(textureSize(shadow_map, 0))).rg;
            sum_w += c;
        }
    }
    vec2 data = sum / sum_w;

    float mu = data.r;
    float z = shadow_pos.z - bias;
    float m2 = data.g;
    float sigma = m2 - mu * mu;
    float shadow_factor = (z < mu) ? 1.0 : sigma / (sigma + (z - mu) * (z - mu));
    float delta = 0.1;
    return (shadow_factor < delta) ? 0.0 : shadow_factor * (1 + delta) - delta;
}

void main()
{
    vec3 texcolor = texture(tex, uv).xyz;

    vec3 normal = raw_normal;
    if (has_normal_map) {
        normal = (normalize(texture(normal_map, uv) * 2.f - 1.f)).xyz;
        normal = normalize((model_ * vec4(normal, 0.0)).xyz);
    }

    float shadow_factor = get_shadow_factor(position);
    vec3 shadowing = light_color * max(0.0, dot(normal, light_direction)) * shadow_factor;

    vec3 reflected = 2.0 * normal * dot(normal, light_direction) - light_direction;

    vec4 roughness = vec4(0.0);
    if (has_specular_map) {
        roughness = texture(specular_map, uv);
    }
    float specular_light = pow(max(0.0, dot(reflected, normalize(camera_position - position))), 4.0) * (roughness.x * roughness.w);

    vec3 diffuse_coef = vec3(0.0);
    if (has_diffuse_map) {
        diffuse_coef = texture(diffuse_map, uv).xyz;
    }
    vec3 diffuse = diffuse_color * light_color * max(0.0, dot(normal , light_direction)) * diffuse_coef;

    vec3 ambient = ambient_color * albedo;
    vec3 specular = specular_light * texcolor;

    vec3 s1 = single_lighter(light_position[0], light_colors[0], light_attenuation[0], normal);
    vec3 s2 = single_lighter(light_position[1], light_colors[1], light_attenuation[1], normal);
    vec3 s3 = single_lighter(light_position[2], light_colors[2], light_attenuation[2], normal);

    vec3 color = (ambient + shadowing * (specular + diffuse) + s1 + s2 + s3) * texcolor;
    vec4 pre_color = vec4(color, 1.0);

    out_color = pre_color / (vec4(1., 1., 1., 0.) + pre_color);
//	out_color = vec4(color, 1.0);
//	out_color = vec4(uv.x, uv.y, 0.0, 1.0);
}
)";

const char debug_vertex_shader_source[] =
R"(#version 330 core

vec2 vertices[6] = vec2[6](
	vec2(-1.0, -1.0),
	vec2( 1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0,  1.0)
);

out vec2 texcoord;

void main()
{
	vec2 position = vertices[gl_VertexID];
	gl_Position = vec4(position * 0.25 + vec2(-0.75, -0.75), 0.0, 1.0);
	texcoord = position * 0.5 + vec2(0.5);
}
)";

const char debug_fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D shadow_map;

in vec2 texcoord;

layout (location = 0) out vec4 out_color;

void main()
{
	out_color = texture(shadow_map, texcoord);
}
)";

const char shadow_vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;

void main()
{
	gl_Position = transform * model * vec4(in_position, 1.0);
}
)";

const char shadow_fragment_shader_source[] =
        R"(#version 330 core

layout (location = 0) out vec4 out_color;

void main()
{
    float z = gl_FragCoord.z;
    out_color = vec4(z, z*z + 0.25 * (dFdx(z) * dFdx(z) + dFdy(z) * dFdy(z)), 0.0, 0.0);
}
)";

const char reflection_vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

layout (location = 0) in vec3 in_position;

void main()
{
	gl_Position = projection * view * model * vec4(in_position, 1.0);
}
)";

const char reflection_fragment_shader_source[] =
        R"(#version 330 core


layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(1.0);
}
)";

//const char reflection_vertex_shader_source[] =
//        R"(#version 330 core
//
//uniform mat4 model;
//uniform mat4 projection;
//uniform mat4 view;
//
//layout (location = 0) in vec3 in_position;
//layout (location = 1) in vec3 in_normal;
//
////out vec3 position;
////out vec3 camera_position;
////out vec3 raw_normal;
////out mat4 model_;
//
//void main()
//{
//	gl_Position = projection * view * model * vec4(in_position, 1.0);
////    position = (model * vec4(in_position, 1.0)).xyz;
////    camera_position = (inverse(view) * vec4(0, 0, 0, 1)).xyz;
////    raw_normal = normalize((model * vec4(in_normal, 0.0)).xyz);
////    model_ = model;
//}
//)";
//
//const char reflection_fragment_shader_source[] =
//        R"(#version 330 core
//
//uniform samplerCube cube_texture;
//
//layout (location = 0) out vec4 out_color;
//
////in vec3 position;
////in vec3 camera_position;
////in vec3 normal;
////in mat4 model;
//
//void main()
//{
////    vec3 I = normalize(camera_position - position);
////    vec3 reflect_normal = (model * vec4(normal, 0.0)).xyz;
////    vec3 reflection = -reflect(I, reflect_normal);
////    vec3 coords = normalize(reflection);
////    out_color = vec4(texture(cube_texture, coords).xyz, 1.0);
//    out_color = vec4(1.0);
//}
//)";

GLuint create_shader(GLenum type, const char * source)
{
	GLuint result = glCreateShader(type);
	glShaderSource(result, 1, &source, nullptr);
	glCompileShader(result);
	GLint status;
	glGetShaderiv(result, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint info_log_length;
		glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
		std::string info_log(info_log_length, '\0');
		glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
		throw std::runtime_error("Shader compilation failed: " + info_log);
	}
	return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
	GLuint result = glCreateProgram();
	glAttachShader(result, vertex_shader);
	glAttachShader(result, fragment_shader);
	glLinkProgram(result);

	GLint status;
	glGetProgramiv(result, GL_LINK_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint info_log_length;
		glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
		std::string info_log(info_log_length, '\0');
		glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
		throw std::runtime_error("Program linkage failed: " + info_log);
	}

	return result;
}

struct vertex
{
	glm::vec3 position;
	glm::vec3 normal;
    glm::vec2 uv;
};

std::pair<std::vector<vertex>, std::vector<std::uint32_t>> parse_obj(std::istream & input)
{
	std::vector<vertex> vertices;
	std::vector<std::uint32_t> indices;
    std::string mtl_filename;

	for (std::string line; std::getline(input, line);)
	{
		std::istringstream line_stream(line);

		char type;
		line_stream >> type;

		if (type == '#')
			continue;

		if (type == 'v')
		{
			vertex v;
			line_stream >> v.position.x >> v.position.y >> v.position.z;
			vertices.push_back(v);
			continue;
		}

		if (type == 'f')
		{
			std::uint32_t i0, i1, i2;
			line_stream >> i0 >> i1 >> i2;
			--i0;
			--i1;
			--i2;
			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);
			continue;
		}

		throw std::runtime_error("Unknown OBJ row type: " + std::string(1, type));
	}

	return {vertices, indices};
}

std::pair<glm::vec3, glm::vec3> bbox(std::vector<vertex> const & vertices)
{
	static const float inf = std::numeric_limits<float>::infinity();

	glm::vec3 min = glm::vec3( inf);
	glm::vec3 max = glm::vec3(-inf);

	for (auto const & v : vertices)
	{
		min = glm::min(min, v.position);
		max = glm::max(max, v.position);
	}

	return {min, max};
}

//glm::mat4 move_to_bbox(glm::vec3 minCur, glm::vec3 maxCur, glm::vec3 min, glm::vec3 max) {
//    // TODO Переносим сцену в нужные координаты
//    float cx = (min.x - max.x) / (maxCur.x - minCur.x);
//    float cy = (min.y - max.y) / (maxCur.y - minCur.y);
//    float cz = (min.z - max.z) / (maxCur.z - minCur.z);
//
//    glm::mat4 transform(1.f);
//    transform[0][0] = cx;
//    transform[1][1] = cy;
//    transform[2][2] = cz;
//    transform[3][0] = min.x - minCur.x;
//    transform[3][1] = min.y - minCur.y;
//    transform[3][2] = min.z - minCur.z;
//
//    return transform;
//}

void add_ground_plane(std::vector<vertex> & vertices, std::vector<std::uint32_t> & indices)
{
	auto [ min, max ] = bbox(vertices);

	glm::vec3 center = (min + max) / 2.f;
	glm::vec3 size = (max - min);
	size.x = size.z = std::max(size.x, size.z);

	float W = 5.f;
	float H = 0.5f;

	vertex v0, v1, v2, v3;
	v0.position = {center.x - W * size.x, center.y - H * size.y, center.z - W * size.z};
	v1.position = {center.x - W * size.x, center.y - H * size.y, center.z + W * size.z};
	v2.position = {center.x + W * size.x, center.y - H * size.y, center.z - W * size.z};
	v3.position = {center.x + W * size.x, center.y - H * size.y, center.z + W * size.z};

	std::uint32_t base_index = vertices.size();
	vertices.push_back(v0);
	vertices.push_back(v1);
	vertices.push_back(v2);
	vertices.push_back(v3);

	indices.push_back(base_index + 0);
	indices.push_back(base_index + 1);
	indices.push_back(base_index + 2);
	indices.push_back(base_index + 2);
	indices.push_back(base_index + 1);
	indices.push_back(base_index + 3);
}

void fill_normals(std::vector<vertex> & vertices, std::vector<std::uint32_t> const & indices)
{
	for (auto & v : vertices)
		v.normal = glm::vec3(0.f);

	for (std::size_t i = 0; i < indices.size(); i += 3)
	{
		auto & v0 = vertices[indices[i + 0]];
		auto & v1 = vertices[indices[i + 1]];
		auto & v2 = vertices[indices[i + 2]];

		glm::vec3 n = glm::cross(v1.position - v0.position, v2.position - v0.position);
		v0.normal += n;
		v1.normal += n;
		v2.normal += n;
	}

	for (auto & v : vertices)
		v.normal = glm::normalize(v.normal);
}

class ShadowShaderProgram {
public:
    GLuint program, model_location, transform_location;
    ShadowShaderProgram (GLuint program) : program(program){
        model_location = glGetUniformLocation(program, "model");
        transform_location = glGetUniformLocation(program, "transform");
    }

    void set_params(glm::mat4 & model, glm::mat4 & transform) {
        glUseProgram(program);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));
    }
};

struct SceneShaderLocations {
    GLuint model_location, view_location, projection_location,
            transform_location, ambient_color_location, light_direction_location,
            light_color_location, shadow_map_location, tex_location, albedo_location,
            has_specular_map_location, has_diffuse_map_location, has_normal_map_location,
            specular_map_location, diffuse_map_location, normal_map_location,
            diffuse_color_location, light_pos_loc[3], light_col_loc[3], light_att_loc[3];
};

struct SceneShaderParamsExceptTex {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 transform;
    glm::vec3 ambient;
    glm::vec3 light_direction;
    glm::vec3 light_color;
    glm::vec3 albedo;


};

class SceneShaderProgram {
public:
    GLuint program;
    SceneShaderLocations locations;

    SceneShaderProgram(GLuint program) : program(program) {
        locations.model_location = glGetUniformLocation(program, "model");
        locations.view_location = glGetUniformLocation(program, "view");
        locations.projection_location = glGetUniformLocation(program, "projection");
        locations.transform_location = glGetUniformLocation(program, "transform");

        locations.ambient_color_location = glGetUniformLocation(program, "ambient_color");
        locations.diffuse_color_location = glGetUniformLocation(program, "diffuse_color");
        locations.albedo_location = glGetUniformLocation(program, "albedo");
        locations.light_direction_location = glGetUniformLocation(program, "light_direction");
        locations.light_color_location = glGetUniformLocation(program, "light_color");

        locations.has_specular_map_location = glGetUniformLocation(program, "has_specular_map");
        locations.has_diffuse_map_location = glGetUniformLocation(program, "has_diffuse_map");
        locations.has_normal_map_location = glGetUniformLocation(program, "has_normal_map");

        locations.shadow_map_location = glGetUniformLocation(program, "shadow_map");
        locations.tex_location = glGetUniformLocation(program, "tex");
        locations.specular_map_location = glGetUniformLocation(program, "specular_map");
        locations.diffuse_map_location = glGetUniformLocation(program, "diffuse_map");
        locations.normal_map_location = glGetUniformLocation(program, "normal_map");

        for (int i = 0; i < 3; i++) {
            std::stringstream pos_str, col_str, att_str;
            pos_str << "light_position[" << i << "]";
            col_str << "light_colors[" << i << "]";
            att_str << "light_attenuation[" << i << "]";
            locations.light_pos_loc[i] = glGetUniformLocation(program, pos_str.str().c_str());
            locations.light_col_loc[i] = glGetUniformLocation(program, col_str.str().c_str());
            locations.light_att_loc[i] = glGetUniformLocation(program, att_str.str().c_str());
        }
    }

    void set_all_except_tex(SceneShaderParamsExceptTex & p) {
        glUseProgram(program);
        glUniformMatrix4fv(locations.model_location, 1, GL_FALSE, reinterpret_cast<float *>(&p.model));
        glUniformMatrix4fv(locations.view_location, 1, GL_FALSE, reinterpret_cast<float *>(&p.view));
        glUniformMatrix4fv(locations.projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&p.projection));
        glUniformMatrix4fv(locations.transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&p.transform));

        glUniform3fv(locations.ambient_color_location, 1, reinterpret_cast<float *>(&p.ambient));
        glUniform3fv(locations.albedo_location, 1, reinterpret_cast<float *>(&p.albedo));
        glUniform3fv(locations.light_direction_location, 1, reinterpret_cast<float *>(&p.light_direction));
        glUniform3fv(locations.light_color_location, 1, reinterpret_cast<float *>(&p.light_color));
    }

    void set_shadow_map(GLint v) {
        glUseProgram(program);
        glUniform1i(locations.shadow_map_location, v);
    }

    void set_tex(GLint v) {
        glUseProgram(program);
        glUniform1i(locations.tex_location, v);
    }

    void set_single_lighter(std::vector<glm::vec3> & positions, std::vector<glm::vec3> & colors) {
        for (int i = 0; i < 3; i++) {
            glUniform3f(locations.light_pos_loc[i], positions[i].x, positions[i].y, positions[i].z);
            glUniform3f(locations.light_col_loc[i], colors[i].x, colors[i].y, colors[i].z);
            glUniform3f(locations.light_att_loc[i], 1.0, 0.0, 0.1);
        }
    }
};


class Texture {
public:
    GLuint id;
    int width, height, n;
    bool empty;

    Texture() {
        id = 0;
        width = 0;
        height = 0;
        n = 0;
        empty = true;
    }

    Texture(const char *filename) {

        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        unsigned char * data = stbi_load(filename, &width, &height, &n, 0);

        if (data) {
            if (n == 3)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            else
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        } else {
            throw std::runtime_error("Failed to load texture");
        }
        stbi_image_free(data);

        empty = false;
    }

    bool ready() {
        return !empty;
    }
};


class Material {
public:
    std::string name = "";

    glm::vec3 ambient = {1.f, 1.f, 1.f};
    glm::vec3 diffuse = {1.f, 1.f, 1.f};
    glm::vec3 specular = {0.f, 0.f, 0.f};
    float spec_exp = 0.f;
    float transparency = 0.f;
    float optical_dencity = 1.f;
    int illum = 2;

    Texture tex;
    Texture diffuse_map;
    Texture specular_map;
    Texture normal_map;

    Material() = default;

    explicit Material(std::string mtl_name) {
        name = std::move(mtl_name);
    }

    void load_texture(const char* path) {
        tex = Texture(path);
    }

    void load_diffuse_map(const char* path) {
        diffuse_map = Texture(path);
    }

    void load_specular_map(const char* path) {
        specular_map = Texture(path);
    }

    void load_normal_map(const char* path) {
        normal_map = Texture(path);
    }

    bool has_texture() {
        return tex.ready();
    }

    bool has_diffuse_map() {
        return diffuse_map.ready();
    }

    bool has_specular_map() {
        return specular_map.ready();
    }

    bool has_normal_map() {
        return normal_map.ready();
    }

};

Material DEFAULT_MATERIAL = Material();


class Object {
public:
    std::vector<vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::pair<glm::vec3, glm::vec3> bounding_box;

    std::string name;

    GLuint vao, vbo, ebo;

    Material& material;

    bool ready_to_draw = false, ready_data = false;

    Object() = default;

    Object(std::string & name, std::vector<glm::vec3> & positions,
           std::vector<glm::vec3> & normals, std::vector<glm::vec2> & uvs,
           std::vector<triplet> & ind, Material & mat) : material(mat) {

        fill_data(name, positions, normals, uvs, ind);
        set_bbox();

    }

    Object(std::vector<vertex> & vertices, std::vector<std::uint32_t> & indices) : material(DEFAULT_MATERIAL) {
        this->vertices = std::vector<vertex>(vertices);
        for (unsigned int & indice : indices) {
            this->indices.push_back(indice);
        }
    }

    void set_bbox() {
        bounding_box = bbox(vertices);
    }

    std::pair<glm::vec3, glm::vec3> get_bbox() {
        return bounding_box;
    }

    void fill_data (std::string & obj_name, std::vector<glm::vec3> & positions,
                    std::vector<glm::vec3> & normals, std::vector<glm::vec2> & uvs, std::vector<triplet> & ind) {

        std::unordered_map<triplet, std::uint32_t, TripletHasher> index_matching;

        name = obj_name;

        for (auto & it : ind) {
            if (index_matching.contains(it)) {
                indices.push_back(index_matching[it]);
                continue;
            }

            std::uint32_t c = it.at(0), tc = it.at(1), nc = it.at(2);

            vertex v{};
            v.position = positions[c];
            v.normal = normals[nc];
            v.uv = uvs[tc];

            index_matching[it] = vertices.size();
            indices.push_back(vertices.size());
            vertices.push_back(v);
        }

        ready_data = true;

    }

    void fill_buffers() {

        if (!ready_data) return;

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(sizeof(vertices[0].position)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(sizeof(vertices[0].position) + sizeof(vertices[0].normal)));

        ready_to_draw = true;

    }

    bool drawable() {
        return ready_to_draw && ready_data;
    }

    void draw() {
        if (!drawable()) {
            throw std::runtime_error("Object " + name + " is not ready to draw");
        }

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);

    }

    void draw_tex(SceneShaderProgram & program) {
        if (!drawable()) {
            throw std::runtime_error("Object " + name + " is not ready to draw");
        }

        if (!material.has_texture()) {
            draw();
            return;
        }

        glUseProgram(program.program);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, material.tex.id);

        if (material.has_specular_map()) {
            glUniform1i(program.locations.has_specular_map_location, true);
            glUniform1i(program.locations.specular_map_location, 2);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, material.specular_map.id);
        } else {
            glUniform1i(program.locations.has_specular_map_location, false);
        }

        glUniform3fv(program.locations.diffuse_color_location, 1, reinterpret_cast<float *>(&material.diffuse));

        if (material.has_diffuse_map()) {
            glUniform1i(program.locations.has_diffuse_map_location, true);
            glUniform1i(program.locations.diffuse_map_location, 3);

            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, material.diffuse_map.id);
        } else {
            glUniform1i(program.locations.has_diffuse_map_location, false);
        }

        if (material.has_normal_map()) {
            glUniform1i(program.locations.has_normal_map_location, true);
            glUniform1i(program.locations.normal_map_location, 4);

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, material.normal_map.id);
        } else {
            glUniform1i(program.locations.has_normal_map_location, false);
        }

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);

    }
};

class ReflectionDrawer {
public:
    GLuint fbo = 0;
    GLuint cubemap_texture = 0;
    GLsizei cubemap_resolution;

    ReflectionDrawer(GLsizei resolution) : cubemap_resolution(resolution) {
        glGenTextures(1, &cubemap_texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture);

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        for(int i = 0; i < 6; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA8, resolution, resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }

        glGenFramebuffers(1, &fbo);
        GLuint cubemap_renderbuffer;
        glGenRenderbuffers(1, &cubemap_renderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, cubemap_renderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, resolution, resolution);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, cubemap_renderbuffer);

        if(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer error");
        }
    }
};

class ShadowDrawer {
public:
    GLuint fbo = 0;
    GLuint shadow_map = 0;
    GLuint depth_map = 0;
    GLsizei shadow_map_resolution;

    ShadowDrawer(GLsizei resolution) : shadow_map_resolution(resolution) {

        glGenTextures(1, &shadow_map);
        glBindTexture(GL_TEXTURE_2D, shadow_map);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, shadow_map_resolution, shadow_map_resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

        glGenTextures(1, &depth_map);
        glBindTexture(GL_TEXTURE_2D, depth_map);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_map_resolution, shadow_map_resolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadow_map, 0);
        glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_map, 0);
        if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("Incomplete framebuffer!");
    }

    glm::mat4 get_shadow_transform(std::pair<glm::vec3, glm::vec3> box, std::vector<glm::vec3*> & light_xyz) {

        auto [minV, maxV] = box;
        glm::vec3 C = (minV + maxV) * 0.5f;

        glm::vec3 diffV[4] = {
                glm::vec3(0, 0, 0),
                glm::vec3(maxV.x - minV.x, 0, 0),
                glm::vec3(0, maxV.y - minV.y, 0),
                glm::vec3(0, 0, maxV.z - minV.z)
        };

        float maxMod = 0, scP;

        for (auto v: light_xyz) {
            maxMod = 0;
            for (auto diff: diffV) {
                glm::vec3 m0 = minV + diff;
                scP = abs(glm::dot(m0 - C, *v));
                if (scP > maxMod) {
                    maxMod = scP;
                }

                glm::vec3 m1 = maxV - diff;
                scP = abs(glm::dot(m1 - C, *v));
                if (scP > maxMod) {
                    maxMod = scP;
                }
            }

            float l = glm::length(*v);

            *v = *v / l * maxMod;
        }

        glm::mat4 transform = glm::mat4(1.f);

        transform[0] = glm::vec4(*light_xyz[0], 0.f);
        transform[1] = glm::vec4(*light_xyz[1], 0.f);
        transform[2] = glm::vec4(*light_xyz[2], 0.f);
        transform[3] = glm::vec4(C, 1.f);

        return glm::inverse(transform);
    }
};


class Scene {
public:
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<triplet> indices;

    std::unordered_map<std::string, Material> materials;

    std::pair<glm::vec3, glm::vec3> bounding_box;
    glm::mat4 transform;

    std::vector<Object> objects;

    std::string path;
    std::string current_mtl_name;
    std::string cur_obj_name;

    ShadowDrawer sd;

    Scene(std::string scene_lib_path, GLsizei shadow_map_resolution = 1024) : sd(1024) {
        path = std::move(scene_lib_path);
    }

    void load_obj(std::istream & input) {
        std::vector<vertex> v;
        std::vector<std::uint32_t> i;
        std::tie(v, i) = parse_obj(input);
        Object obj(v, i);
        objects.push_back(obj);
    }

    void parse_scene(const char *filename) {

        std::ifstream input(path + filename);

        bool prevF = false;

        for (std::string line; std::getline(input, line);)
        {
            std::istringstream line_stream(line);

            std::string word;
            line_stream >> word;

            if (prevF) {
                if (word == "#" || word == "usemtl") {
                    std::cout << "Shit";
                    bool clean_vertices = false;
                    if (word == "#") clean_vertices = true;
                    create_object();
                    clean_indices(clean_vertices);
                }
            }
            prevF = false;

            if (word.empty() || word == "#" || word == "s") continue;

            if (word == "mtllib") {
                std::string mtl_filename;
                line_stream >> mtl_filename;

                parse_mtl((path + mtl_filename).c_str());
                continue;
            }

            if (word == "usemtl") {
                std::string mtl_name;
                line_stream >> current_mtl_name;

                continue;
            }

            if (word == "v") {
                glm::vec3 v;
                line_stream >> v.x >> v.y >> v.z;
                v.x /= 2000;
                v.y /= 2000;
                v.z /= 2000;
                vertices.push_back(v);
                continue;
            }

            if (word == "vn") {
                glm::vec3 n;
                line_stream >> n.x >> n.y >> n.z;
                normals.push_back(n);
                continue;
            }

            if (word == "vt") {
                glm::vec2 uv;
                float shit;
                line_stream >> uv.x >> uv.y >> shit;
                uvs.push_back(uv);
                continue;
            }

            if (word == "f") {

                prevF = true;

                std::string token;
                std::vector<triplet> cur_ind;

                while (line_stream >> token) {
                    cur_ind.push_back(getCoordsFromToken(token));
                }
                if (cur_ind.size() == 4) {
                    triplet tmp = cur_ind.back();
                    cur_ind.pop_back();
                    cur_ind.push_back(cur_ind[0]);
                    cur_ind.push_back(cur_ind[2]);
                    cur_ind.push_back(tmp);
                }

                indices.insert(indices.end(), std::make_move_iterator(cur_ind.begin()), std::make_move_iterator(cur_ind.end()));
                continue;
            }

            if (word == "g") {
                line_stream >> cur_obj_name;
                continue;
            }

            throw std::runtime_error("Unknown OBJ row type: " + word);

        }
    }

    void parse_mtl(const char * filename) {
        std::string curmtl_name;

        std::ifstream input(filename);

        for (std::string line; std::getline(input, line);) {

            std::istringstream line_stream(line);

            std::string word;
            line_stream >> word;

            if (word.empty() || word == "#") continue;
            if (word == "d" || word == "illum") continue;

            if (word == "newmtl") {

                std::string mtl_name;
                line_stream >> mtl_name;

                curmtl_name = mtl_name;

                materials.insert({mtl_name, Material(mtl_name)});
                continue;
            }

            if (word == "Ka") {
                glm::vec3 amb;
                line_stream >> amb.x >> amb.y >> amb.z;

                materials[curmtl_name].ambient = amb;
                continue;
            }

            if (word == "Kd") {
                glm::vec3 diff;
                line_stream >> diff.x >> diff.y >> diff.z;

                materials[curmtl_name].diffuse = diff;
                continue;
            }

            if (word == "Ks") {
                glm::vec3 spec;
                line_stream >> spec.x >> spec.y >> spec.z;

                materials[curmtl_name].specular = spec;
                continue;
            }

            if (word == "Ns") {
                float spexp;
                line_stream >> spexp;

                materials[curmtl_name].spec_exp = spexp;
                continue;
            }

            if (word == "Tr") {
                float tr;
                line_stream >> tr;

                materials[curmtl_name].transparency = tr;
                continue;
            }

            if (word == "Ni") {
                float tr;
                line_stream >> tr;

                materials[curmtl_name].optical_dencity = tr;
                continue;
            }

            if (word == "map_Ka") {
                std::string s;
                line_stream >> s;

                materials[curmtl_name].load_texture((path + s).c_str());
                continue;
            }

            if (word == "map_Kd") {
                std::string s;
                line_stream >> s;

                materials[curmtl_name].load_diffuse_map((path + s).c_str());
                continue;
            }

            if (word == "map_Ks") {
                std::string s;
                line_stream >> s;

                materials[curmtl_name].load_specular_map((path + s).c_str());
                continue;
            }

            if (word == "norm") {
                std::string s;
                line_stream >> s;

                materials[curmtl_name].load_normal_map((path + s).c_str());
                continue;
            }
        }
    }

    void create_object() {
        std::cout << "Create object\n";
        Object obj(cur_obj_name, vertices, normals, uvs, indices, materials[current_mtl_name]);
        obj.fill_buffers();
        objects.push_back(obj);
        std::cout << "Object created\n";
    }

    void draw_shadows(glm::mat4 & model, glm::vec3 & light_direction, ShadowShaderProgram & shadow_program) {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, sd.fbo);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, sd.shadow_map_resolution, sd.shadow_map_resolution);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glm::vec3 light_z = -light_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, {0.f, 1.f, 0.f}));
        glm::vec3 light_y = glm::cross(light_x, light_z);

        std::vector<glm::vec3*> light_xyz = {
                &light_x,
                &light_y,
                &light_z

        };

        transform = sd.get_shadow_transform(bounding_box, light_xyz);

        shadow_program.set_params(model, transform);
        glUseProgram(shadow_program.program);

        draw();
    }

    void draw_scene(int width, int height, SceneShaderParamsExceptTex & params,
                    SceneShaderProgram & program) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sd.shadow_map);
        glGenerateMipmap(GL_TEXTURE_2D);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);

        glClearColor(0.8f, 0.8f, 0.9f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        program.set_all_except_tex(params);
        glUseProgram(program.program);

        draw_tex(program);
    }

    void draw() {
        for (auto & object : objects) {
            object.draw();
        }
    }

    void draw_tex(SceneShaderProgram & program) {

        std::vector<glm::vec3> pos = {
                {-0.3, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {0.3, 0.0, 0.0},
        };
        std::vector<glm::vec3> colors = {
                {1.f, 0.f, 0.f},
                {0.f, 1.f, 0.f},
                {0.f, 0.f, 1.f}
        };

        program.set_single_lighter(pos, colors);

        for (auto & object : objects) {
            object.draw_tex(program);
        }
    }

    void set_bounding_box() {
        std::vector<vertex> bboxes;
        for (auto & object : objects) {
            vertex v1{}, v2{};
            auto [min, max] = object.get_bbox();
            v1.position = min;
            v2.position = max;
            bboxes.push_back(v1);
            bboxes.push_back(v2);
        }
        bounding_box = bbox(bboxes);
    }

    std::pair<glm::vec3, glm::vec3> get_bounding_box() {
        return bounding_box;
    }

private:

    std::uint32_t last_used_index[3] = {0, 0, 0};

    void clean_indices(bool clean_vertices) {
        indices.clear();
        if (clean_vertices) {
            last_used_index[0] += vertices.size();
            vertices.clear();
            last_used_index[1] += uvs.size();
            uvs.clear();
            last_used_index[2] += normals.size();
            normals.clear();
        }
    }

    triplet getCoordsFromToken(std::string & token) {
        triplet coords;
        std::string item;
        std::stringstream s(token);

        for (int i = 0; i<3; i++) {
            getline(s, item, '/');
            coords[i] = std::stoi(item) - 1;
            coords[i] -= last_used_index[i];
        }

        return coords;
    }

};

class ReflectiveObject {
public:
    std::vector<vertex> vertices;
    std::vector<std::uint32_t> indices;
    GLuint vao=0, vbo=0, ebo=0;
    GLuint program=0;
    GLuint model_location=0, projection_location=0, view_location=0, cube_texture_location=0;

    glm::mat4 view;

    glm::mat4 model;
    float diff = 0, coef = 0.0000001, trans = 0;

    ReflectionDrawer rd;

    ReflectiveObject() : rd(256) {}

    void load_object(std::istream & input) {
        std::tie(vertices, indices) = parse_obj(input);
        fill_normals(vertices, indices);

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(sizeof(vertices[0].position)));

        model = glm::mat4(1.f);
        view = glm::mat4(1.f);

        rd = ReflectionDrawer(1024);
    }

    void set_program(GLuint program) {
        this->program = program;
        model_location = glGetUniformLocation(program, "model");
        projection_location = glGetUniformLocation(program, "projection");
        view_location = glGetUniformLocation(program, "view");
//        cube_texture_location = glGetUniformLocation(program, "cube_texture");
    }

//    void draw(glm::mat4 & projection, glm::mat4 & view, float time, Scene& s, SceneShaderProgram & main_program, SceneShaderParamsExceptTex params, int width, int height) {
//
//        glm::vec3 trans = {cos(time)*0.3, 0, 0};
//
//        model = glm::translate(glm::mat4(1.f), trans);
//
//        std::vector<glm::mat4> views = {
//                glm::translate(glm::rotate(view,1 * glm::pi<float>() / 2.f, {0.f, 1.f, 0.f}), trans),
//                glm::translate(glm::rotate(view, 3 * glm::pi<float>() / 2.f, {0.f, 1.f, 0.f}), trans),
//                glm::translate(glm::rotate(view, -glm::pi<float>() / 2.f, {1.f, 0.f, 0.f}), trans),
//                glm::translate(glm::rotate(view, glm::pi<float>() / 2.f, {1.f, 0.f, 0.f}), trans),
//                glm::translate(glm::rotate(view, 2 * glm::pi<float>() / 2.f, {0.f, 1.f, 0.f}), trans),
//                glm::translate(view, trans)
//        };
//
//        glActiveTexture(GL_TEXTURE0);
//        glBindTexture(GL_TEXTURE_2D, s.sd.shadow_map);
//        glActiveTexture(GL_TEXTURE0 + 5);
//        glBindTexture(GL_TEXTURE_CUBE_MAP, rd.cubemap_texture);
//
//        for (int i = 0; i<6; i++) {
//            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rd.fbo);
//            glViewport(0, 0, rd.cubemap_resolution, rd.cubemap_resolution);
//            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, rd.cubemap_texture, 0);
//
//            params.view = views[i];
//            main_program.set_all_except_tex(params);
//            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//            s.draw_tex(main_program);
//        }
//
//        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
//        glViewport(0, 0, width, height);
//        glUseProgram(program);
//
//        glActiveTexture(rd.cubemap_texture);
//        glBindTexture(GL_TEXTURE_CUBE_MAP, rd.cubemap_texture);
//
//        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
//        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
//        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
//        glUniform1i(cube_texture_location, 5);
//
//
//        glBindVertexArray(vao);
//        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
//    }

    void draw_simple(glm::mat4 & projection, glm::mat4 & view, float time) {
        glm::vec3 trans = {cos(time)*0.3, 0, 0};

        model = glm::translate(glm::mat4(1.f), trans);

        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform1i(cube_texture_location, 5);


        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
    }

//    void step(glm::vec3 translate) {
//        model = glm::translate(model, {trans, 0, 0});
//
//        if (trans > 1.0 || trans < -1.0) coef *= -1;
//    }
};

int main() try
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		sdl2_fail("SDL_Init: ");

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window * window = SDL_CreateWindow("Graphics course practice 9",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		800, 600,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

	if (!window)
		sdl2_fail("SDL_CreateWindow: ");

	int width, height;
	SDL_GetWindowSize(window, &width, &height);

	SDL_GLContext gl_context = SDL_GL_CreateContext(window);
	if (!gl_context)
		sdl2_fail("SDL_GL_CreateContext: ");

	if (auto result = glewInit(); result != GLEW_NO_ERROR)
		glew_fail("glewInit: ", result);

	if (!GLEW_VERSION_3_3)
		throw std::runtime_error("OpenGL 3.3 is not supported");

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);

    SceneShaderProgram main_program(program);

	glUseProgram(main_program.program);
    main_program.set_shadow_map(0);
    main_program.set_tex(1);

	auto debug_vertex_shader = create_shader(GL_VERTEX_SHADER, debug_vertex_shader_source);
	auto debug_fragment_shader = create_shader(GL_FRAGMENT_SHADER, debug_fragment_shader_source);
	auto debug_program = create_program(debug_vertex_shader, debug_fragment_shader);

	GLuint debug_shadow_map_location = glGetUniformLocation(debug_program, "shadow_map");

	glUseProgram(debug_program);
	glUniform1i(debug_shadow_map_location, 0);

	auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
	auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
	auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);
    ShadowShaderProgram sp(shadow_program);

    Scene sponza(PRACTICE_SOURCE_DIRECTORY "/sponza/");
    {
        sponza.parse_scene("sponza.obj");
//        sponza.parse_scene("test.obj");
    }

    sponza.set_bounding_box();


    ReflectiveObject bunny;
    {
		std::ifstream bunny_file(PRACTICE_SOURCE_DIRECTORY "/bunny.obj");
        bunny.load_object(bunny_file);
	}

    auto reflect_vertex_shader = create_shader(GL_VERTEX_SHADER, reflection_vertex_shader_source);
    auto reflect_fragment_shader = create_shader(GL_FRAGMENT_SHADER, reflection_fragment_shader_source);
    auto reflect_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

    bunny.set_program(reflect_program);


	GLuint debug_vao;
	glGenVertexArrays(1, &debug_vao);

	GLsizei shadow_map_resolution = 1024;

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	float time = 0.f;
	bool paused = false;

	std::map<SDL_Keycode, bool> button_down;

	float view_elevation = glm::radians(45.f);
	float view_azimuth = 4.7544f;
	float camera_distance = 0.5f;
	float camera_target = 0.05f;
	bool running = true;

    float camera_x = 0.f, camera_y = 0.f, camera_z = 0.f;

    SceneShaderParamsExceptTex params;

	while (running)
	{
		for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
		{
		case SDL_QUIT:
			running = false;
			break;
		case SDL_WINDOWEVENT: switch (event.window.event)
			{
			case SDL_WINDOWEVENT_RESIZED:
				width = event.window.data1;
				height = event.window.data2;
				glViewport(0, 0, width, height);
				break;
			}
			break;
		case SDL_KEYDOWN:
			button_down[event.key.keysym.sym] = true;

			if (event.key.keysym.sym == SDLK_SPACE)
				paused = !paused;

			break;
		case SDL_KEYUP:
			button_down[event.key.keysym.sym] = false;
			break;
		}

		if (!running)
			break;

		auto now = std::chrono::high_resolution_clock::now();
		float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
		last_frame_start = now;
		if (!paused)
			time += dt;

		if (button_down[SDLK_j])
            camera_x -= 1.f*dt;
//			camera_distance -= 1.f * dt;
		if (button_down[SDLK_l])
            camera_x += 1.f*dt;
//			camera_distance += 1.f * dt;
        if (button_down[SDLK_i])
            camera_y -= 1.f*dt;
//            view_azimuth -= 1.f * dt;
        if (button_down[SDLK_k])
            camera_y += 1.f*dt;
//            view_azimuth += 1.f * dt;
        if (button_down[SDLK_u])
            camera_z -= 1.f*dt;
//            view_elevation -= 1.f * dt;
        if (button_down[SDLK_o])
            camera_z += 1.f*dt;
//            view_elevation += 1.f * dt;

		if (button_down[SDLK_a])
			view_azimuth -= 2.f * dt;
		if (button_down[SDLK_d])
			view_azimuth += 2.f * dt;
        if (button_down[SDLK_w])
            view_elevation -= 2.f * dt;
        if (button_down[SDLK_s])
            view_elevation += 2.f * dt;

        params.model = glm::mat4(1.f);
		params.light_direction = glm::normalize(glm::vec3(std::cos(time * 0.5f), 1.f, std::sin(time * 0.5f)));
        sponza.draw_shadows(params.model, params.light_direction, sp);
        params.transform = sponza.transform;

        float near = 0.01f;
		float far = 10.f;

		glm::mat4 view(1.f);
        view = glm::rotate(view, view_elevation, {1.f, 0.f, 0.f});
        view = glm::rotate(view, view_azimuth, {0.f, 1.f, 0.f});
        view = glm::translate(view, {camera_x, camera_y, camera_z});
        params.view = view;

		params.projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);
        params.ambient = {1.f, 1.f, 1.f};
        params.albedo = {0.5f, 0.5f, 0.5f};
        params.light_color = {0.8f, 0.8f, 0.8f};

        sponza.draw_scene(width, height, params, main_program);

//        bunny.draw(params.projection, params.view, time, sponza, main_program, params, width, height);
        bunny.draw_simple(params.projection, params.view, time);

//        glUseProgram(debug_program);
//        glBindTexture(GL_TEXTURE_2D, shadow_map);
//        glBindVertexArray(debug_vao);
//        glDrawArrays(GL_TRIANGLES, 0, 6);

		SDL_GL_SwapWindow(window);

	}

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
