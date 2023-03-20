#version 330

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
uniform mat4 mvp;
uniform vec4 tint;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

out vec2 v_uv;
out vec4 v_tint;

void main() {
    v_uv = in_texcoord_0;
    v_tint = tint;
    gl_Position =  mvp * vec4(in_position, 1.0);
}