library(tidyverse)

#setwd('G:/CCD/CVE/RESPIRATORIAS/_EQUIPE/Denise/modelling/pop_sarampo/8_pastas_auxiliares/estimativa_suscetibilidade/')

df_total <- data.frame()
years <- seq(2019, 2019) # update here the years

for(year in years){
  file_name <- paste0('data/pop-', '2019', '.csv')
  df_aux <- read.csv2(file_name)
  df_aux <- df_aux %>% filter(Municipio != 'Total')
  #df_aux <- df_aux %>% select(!Total)
  df_aux <- df_aux %>% mutate(ID_MUN = substr(Municipio, 1, 6))
  df_aux <- df_aux %>% select(!Municipio)
  df_aux <- df_aux %>% mutate(Ano = year)
  colnames(df_aux) <- c('< 1', paste0('', 1:79), '80 e +', 'ID_MUN', 'Ano')
  df_total <- rbind(df_total, df_aux)
}

#save(df_total, file = 'pop_sarampo.RData')

df_total_2 <- df_total
df_total_2 <- df_total_2 %>%
  mutate(`79` = as.numeric(`79`))
df_total_2 <- df_total_2 %>%
  mutate(`79` = replace_na(`79`, 0))
age_cols <- setdiff(colnames(df_total), c("ID_MUN", "Ano"))
df_total_2 <- df_total_2 %>%
  pivot_longer(
    cols = all_of(age_cols),
    names_to = "idade",
    values_to = "POP"
  )
#save(df_total_2, file = 'pop_sarampo_long.RData')

pop_aux <- df_total_2
pop_aux <- pop_aux %>%
    mutate(idade = ifelse(idade == '< 1', 0, idade)) %>%
    mutate(idade = ifelse(idade == '80 e +', 81, idade))
pop_aux <- pop_aux %>% mutate(idade = as.numeric(idade))
pop_aux <- pop_aux %>%
    mutate(FX_ETARIA = case_when(
        idade < 1 ~ '< 01 ano',
        idade >= 1 & idade < 5 ~ '01 a 04 anos',
        idade >= 5 & idade < 10 ~ '05 a 09 anos',
        idade >= 10 & idade < 15 ~ '10 a 14 anos',
        idade >= 15 & idade < 20 ~ '15 a 19 anos',
        idade >= 20 & idade < 30 ~ '20 a 29 anos',
        idade >= 30 & idade < 40 ~ '30 a 39 anos',
        idade >= 40 & idade < 50 ~ '40 a 49 anos',
        idade >= 50 & idade < 59 ~ '50 a 59 anos',
        idade >= 60 ~ '60 e +'
    ))

write.csv(pop_aux, file = 'pop-sarampo.csv')


