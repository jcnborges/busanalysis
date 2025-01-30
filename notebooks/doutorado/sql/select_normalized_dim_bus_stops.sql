USE busanalysis_dw;

SET @regex = 'Terminal (Capão da Imbuia|Pinheirinho|Portão|Bairro Alto|Barreirinha|Boa Vista|Boqueirão|Cabral|Cachoeira|Caiuá|Campina do Siqueira|Campo Comprido|Capão Raso|Carmo|Centenário|CIC|Fazendinha|Hauer|Maracanã|Oficinas|Pinhais|Santa Cândida|Santa Felicidade|Sítio Cercado|Tatuquara|Guadalupe)';

SELECT DISTINCT
	`legacy_id`,
	`name`,
    CASE WHEN REGEXP_SUBSTR(`name`, @regex) IS NULL THEN `name` ELSE REGEXP_SUBSTR(`name`, @regex) END AS `name_norm`,
    `type`,
    CASE 		
        WHEN `type` = "Plataforma" OR `name` REGEXP @regex THEN "Bus terminal"
        WHEN `type` = "Estação tubo" THEN "Tube station"
        WHEN `type` = "Linha Turismo" THEN "Tourism line"
        WHEN `type` = "Especial Madrugueiro" THEN "Dawn bus"        
        WHEN `type` IN ("Chapéu chinês", "Domus", "Novo mobiliário", "Placa em cano", "Placa em poste", "Sem demarcação") THEN "Street bus stop"
        ELSE "Others"
    END AS `type_norm`
FROM dim_bus_stop