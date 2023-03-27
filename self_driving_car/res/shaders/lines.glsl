#version 330

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 view;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}

#endif

#if defined FRAGMENT_SHADER

out vec4 frag_color;

uniform vec4 color;

void main() {
    frag_color = color;
}

#endif