const { default: mongoose } = require("mongoose");

const express = require("express");
const app = express();
const PORT = 5000;
const { MONGOURI } = require("./keys.js");
mongoose
  .connect("mongodb://localhost:27017/NITR", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => {
    console.log("connected to db");
  });
require("./models/user");
require("./models/post");
app.use(express.json());
// mongoose.model("User");
app.use(require("./routes/auth"));
app.use(require("./routes/post"));
app.use(require("./routes/user"));

app.listen(PORT, () => {
  console.log("connected to port:", PORT);
});
