% Lectura de los patrones de entrenamiento desde un archivo CSV
data_entrenamiento = csvread('XOR_trn.csv');
entradas_entrenamiento = data_entrenamiento(:, 1:end-1);
salidas_deseadas_entrenamiento = data_entrenamiento(:, end);

% Introducir alteraciones aleatorias (< 5%) en los patrones de entrenamiento
alteracion_maxima = 0.02;
entradas_entrenamiento = entradas_entrenamiento + alteracion_maxima * randn(size(entradas_entrenamiento));

% Lectura de los patrones de prueba desde un archivo CSV
data_prueba = csvread('XOR_tst.csv');
entradas_prueba = data_prueba(:, 1:end-1);

% Introducir alteraciones aleatorias (< 5%) en los patrones de prueba
entradas_prueba = entradas_prueba + alteracion_maxima * randn(size(entradas_prueba));


% Parámetros de entrenamiento
criterio_finalizacion = 'error'; % 'error' o 'epocas'
max_epocas = 1000;
tasa_aprendizaje = 0.1;

% Inicialización de pesos y umbral
num_entradas = size(entradas_entrenamiento, 2);
pesos = rand(1, num_entradas);
umbral = rand();

% Entrenamiento del perceptrón
epoca = 1;
error_global = inf;

while (epoca <= max_epocas && error_global > 0)
    error_global = 0;
    
    for i = 1:size(entradas_entrenamiento, 1)
        % Cálculo de la salida del perceptrón
        salida = sum(entradas_entrenamiento(i, :) .* pesos) + umbral;
        salida_binaria = sign(salida);
        
        % Cálculo del error
        error = salidas_deseadas_entrenamiento(i) - salida_binaria;
        
        % Actualización de pesos y umbral
        pesos = pesos + tasa_aprendizaje * error * entradas_entrenamiento(i, :);
        umbral = umbral + tasa_aprendizaje * error;
        
        error_global = error_global + abs(error);
    end
    
    fprintf('Época: %d, Error global: %f\n', epoca, error_global);
    
    epoca = epoca + 1;
end

% Mostrar gráficamente los patrones y la recta separadora
figure;
scatter(entradas_entrenamiento(salidas_deseadas_entrenamiento == 1, 1), ...
        entradas_entrenamiento(salidas_deseadas_entrenamiento == 1, 2), 'O', 'b');
hold on;
scatter(entradas_entrenamiento(salidas_deseadas_entrenamiento == -1, 1), ...
        entradas_entrenamiento(salidas_deseadas_entrenamiento == -1, 2), 'x', 'r');
        
x_vals = linspace(min(entradas_entrenamiento(:, 1)), max(entradas_entrenamiento(:, 1)));
y_vals = -(pesos(1) * x_vals + umbral) / pesos(2);
plot(x_vals, y_vals, 'g', 'LineWidth', 2);

legend('Salida 1', 'Salida -1', 'Recta separadora');
xlabel('Entrada 1');
ylabel('Entrada 2');
title('Patrones y Recta Separadora');

hold off;

% Prueba del perceptrón entrenado en datos de prueba
for i = 1:size(entradas_prueba, 1)
    salida = sum(entradas_prueba(i, :) .* pesos) + umbral;
    salida_binaria = sign(salida);
    
    fprintf('Entradas: %s, Salida: %d\n', mat2str(entradas_prueba(i, :)), salida_binaria);
end
