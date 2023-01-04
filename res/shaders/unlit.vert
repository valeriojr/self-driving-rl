#version 330

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
uniform mat4 mvp;
uniform vec4 tint;

in vec3 in_vert;
in vec2 in_uv;

out vec2 v_uv;
out vec4 v_tint;

void main() {
    v_uv = in_uv;
    v_tint = tint;
    gl_Position =  mvp * vec4(in_vert, 1.0);
}