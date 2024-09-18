CREATE DATABASE proyecto_BEDU;

DROP DATABASE proyecto_BEDU;

SELECT * FROM diabetes;

SHOW COLUMNS FROM diabetes;

-- Hago una copia de la base de datos
CREATE TABLE copia_diabetes LIKE diabetes;

-- Quitamos columnas que no se usarán
ALTER TABLE diabetes
DROP COLUMN `Marcadores geneticos`,
DROP COLUMN	`Autoanticuerpos`,
DROP COLUMN `Factores ambientales`,
DROP COLUMN `Origen etnico`,
DROP COLUMN `Niveles de encimas digestiva`,
DROP COLUMN `Sistomas de inicio temprano`;


ALTER TABLE diabetes
RENAME COLUMN `Antecedentes familiares` TO `antecedentes_familiares`;

ALTER TABLE diabetes
RENAME COLUMN `Niveles de insulina` TO `niveles_insulina`;

ALTER TABLE diabetes
RENAME COLUMN `Edad` TO `edad`;

ALTER TABLE diabetes
RENAME COLUMN `IMC` TO `imc`;

ALTER TABLE diabetes
RENAME COLUMN `Actividad fisica` TO `actividad_fisica`;

ALTER TABLE diabetes
RENAME COLUMN `Abitos dieteticos` TO `habitos_dieteticos`;

ALTER TABLE diabetes
RENAME COLUMN `Presion sanguinia` TO `presion_sanguinea`;

ALTER TABLE diabetes
RENAME COLUMN `Niveles de colesterol` TO `niveles_colesterol`;

ALTER TABLE diabetes
RENAME COLUMN `Talla` TO `talla`;

ALTER TABLE diabetes
RENAME COLUMN `Nivees de glucosa` TO `niveles_glucosa`;

ALTER TABLE diabetes
RENAME COLUMN `Factores socioecomonicos` TO `factores_socioeconomicos`;

ALTER TABLE diabetes
RENAME COLUMN `Tabaquismo` TO `tabaquismo`;

ALTER TABLE diabetes
RENAME COLUMN `Consumo de alcohol` TO `consumo_alcohol`;

ALTER TABLE diabetes
RENAME COLUMN `Tolerancia a la glucosa` TO `tolerancia_glucosa`;

ALTER TABLE diabetes
RENAME COLUMN `Sindrome de ovario poliquístico` TO `sindrome_ovario_poliquistico`;

ALTER TABLE diabetes
RENAME COLUMN `Diabetes gestional` TO `diabetes_gestacional`;

ALTER TABLE diabetes
RENAME COLUMN `Tipo de embarazo` TO `tipo_embarazo`;

ALTER TABLE diabetes
RENAME COLUMN `Aumento de peso en el embarazo` TO `aumento_peso_embarazo`;

ALTER TABLE diabetes
RENAME COLUMN `Salud pancriatica` TO `salud_pancreatica`;

ALTER TABLE diabetes
RENAME COLUMN `Analisis de orina` TO `analisis_orina`;

ALTER TABLE diabetes
RENAME COLUMN `Peso al nacer` TO `peso_nacimiento`;