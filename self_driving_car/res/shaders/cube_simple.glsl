#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;
in vec2 in_tex_coord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 mvp;

out vec3 normal;
out vec2 tex_coord;

void main() {
    normal = in_normal;
    tex_coord = in_tex_coord;

    gl_Position =  mvp * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

in vec3 normal;
in vec2 tex_coord;

out vec4 frag_color;

uniform vec4 color;
uniform sampler2D diffuse_map;

void main() {
    frag_color = color * texture(diffuse_map, tex_coord);
}
#endif