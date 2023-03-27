# Proposta de projeto: Veículo autônomo

## Introdução
O projeto consiste no desenvolvimento de um veículo autônomo que seja capaz de 
navegar num ambiente urbano respeitando leis de trânsito sem que haja programação explicíta delas, a partir de imagens capturadas do ponto de vista do próprio veículo.
Para isso, será utilizado um algoritmo de aprendizado por reforço.
O estado será a imagem da câmera acoplada ao veículo e as ações correspondem ao controle do veículo, como velocidade desejada, direção desejada etc. 
O treinamento do modelo será feito num ambiente virtual e testado em outros simuladores.


## Hardware utilizado
O hardware utilizado para execução do projeto consistirá em:

1. Microcontrolador da família ESP32

2. Câmera RGB

3. Servomotor para a direção

4. Motor DC para a tração

5. LEDs para indicar o estado do veículo e o controle de semáforos

Devido às restrições de tamanho impostas pela natureza do problema (o veículo não pode ser muito grande para que seja viável construir uma maquete para que sejam realizados os testes) deverá ser utilizado o kit de desenvolvimento ESPCAM

## Requisitos
- [ ] Veículo impresso em 3D e maquete para demonstração do funcionamento
- [ ] Sistema operacional Zephyr
- [ ] Dirige sem sair da pista
- [ ] Dirige na faixa correta
- [ ] Respeita semáforos
- [ ] Faz curvas