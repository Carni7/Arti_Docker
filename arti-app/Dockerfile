# Step 1: Use the specified Node.js version
FROM node:20.13.1 AS build

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json files to the working directory
COPY package*.json ./

# Install Angular CLI globally
RUN npm install -g @angular/cli@17.3.7

# Install dependencies
RUN npm install

# Copy the rest of the application files to the container
COPY . .

# Build the Angular application for production
RUN ng build --configuration production

# Step 2: Serve the Angular application using Nginx
FROM nginx:alpine

# Copy the custom nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the /app directory from the build stage to the final image
COPY --from=build /usr/src/app /usr/share/nginx/app

# Copy the build output from the previous step to Nginx's default public directory
COPY --from=build /usr/src/app/dist/arti-app /usr/share/nginx/html

# Expose port 80 to access the application
EXPOSE 80

# Start Nginx when the container starts
CMD ["nginx", "-g", "daemon off;"]
