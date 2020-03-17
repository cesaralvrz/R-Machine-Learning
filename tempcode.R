pressure
str(pressure)
plot(pressure$temperature, pressure$pressure)
title("Temperatura vs Presión")

precip
precip[1:10]
dotchart(precip[1:10])

mtcars$cyl
unique(mtcars$cyl)

tcyl = table(mtcars$cyl)

barplot(tcyl, col = c("blue", "orange", "grey"))
title("Gráficos de Barras", "Número de coches que tienen n cilindros")

mdeaths
ldeaths
fdeaths
maximo = max(mdeaths, ldeaths, fdeaths)
minimo = min(mdeaths, ldeaths, fdeaths)

plot(mdeaths, col = "red", ylim = c(minimo - 100,maximo + 1000), xlab = "Año", ylab = "Nº muertes")
lines(ldeaths, col = "blue", lty = 2)
lines(fdeaths, col = "green", lty = 2, lwd = 3)
legend(1978, 5000, legend = c("mdeaths","ldeaths","fdeaths"), 
       col = c("red","blue","green"), lty = c(1,2,2), lwd = c(1,1,3))
hhh
pr = precip[1:5]
prmenor = pr[pr < 20]
prmayor = pr[pr >= 20]

plot(prmayor, type = "n", ylim = c(min(prmenor), max(prmayor)), xlim = c(1,sum(length(prmayor), length(prmenor))))
text(c(1:length(prmenor)), prmenor, names(prmenor), font = 3, col = "red")
text(c(length(prmenor)+1:length(prmayor)), prmayor, names(prmayor), font = 2, col = "blue")

library(plotly)
plot_ly(mtcars, x = ~wt, y = ~hp, z = ~qsec) %>% add_markers()

# Cambiamos la lista de 0 y 1 de mtcars$am por Automatico y Manual. 
mtcars$am[which(mtcars$am == 0)] <- 'Automatico'
mtcars$am[which(mtcars$am == 1)] <- 'Manual'
mtcars$am <- as.factor(mtcars$am)

plot_ly(mtcars, x = ~wt, y = ~hp, z = ~qsec, color = ~am, colors = c("red","blue")) %>% add_markers()

library(plotly)

# Cambiamos la lista de 0 y 1 de mtcars$am por Automatico y Manual. 
mtcars$am[which(mtcars$am == 0)] <- 'Automatic'
mtcars$am[which(mtcars$am == 1)] <- 'Manual'
mtcars$am <- as.factor(mtcars$am)

plot_ly(mtcars, x = ~wt, y = ~mpg, z = ~qsec, color = ~am, 
        colors = c("blue", "red")) %>% add_lines()

# Podemos añadir puntos y lineas       
plot_ly(mtcars, x = ~wt, y = ~mpg, z = ~qsec, color = ~am, 
        colors = c("blue", "red")) %>% add_lines() %>% add_markers()

plot_ly(z = volcano) %>% add_surface()

plot_ly(mtcars, x = ~cyl, y = ~mpg, color = ~am, colors = c("blue", "red")) %>% add_bars()

plot_ly(z = volcano) %>% add_heatmap()

fact = factor(mtcars$cyl)
fact
plot_ly(mtcars, x = fact) %>% add_histogram()

pr = precip[1:5]
ds = data.frame(
  labels = names(pr),
  values = pr
)
ds
plot_ly(ds, labels = ~labels, values = ~values) %>% add_pie()







