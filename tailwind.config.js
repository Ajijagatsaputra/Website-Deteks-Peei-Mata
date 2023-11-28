/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./templates/**/*.html",
        "./static/src/**/*.js",
        "./node_modules/flowbite/**/*.js"
    ],
    theme: {
        extend: {
            fontFamily: {
                'sans': ['Plus Jakarta Sans', 'Sans-serif']
            }
        },
    },
    plugins: [
        require('flowbite/plugin')({
            charts: true,
        }),
        require('flowbite-typography'),
    ],
}