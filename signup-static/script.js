// SIGNUP FORM HANDLER
document.getElementById("signupForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const fullname = document.getElementById("fullname").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  try {
    const response = await fetch("http://localhost:5000/signup", {
      method: "POST",
      body: new URLSearchParams({
        name: fullname,
        email: email,
        password: password,
      }),
    });

    const result = await response.json();
    alert(result.message);

    if (response.ok && result.status === "success") {
      localStorage.setItem("userEmail", email);
      window.location.href = `http://localhost:8501/?email=${encodeURIComponent(email)}`;
    }
  } catch (error) {
    alert("⚠️ Server connection failed — make sure Flask is running on port 5000");
    console.error(error);
  }
});

// LOGIN FORM HANDLER
document.getElementById("loginForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const email = document.getElementById("loginEmail").value;
  const password = document.getElementById("loginPassword").value;

  try {
    const response = await fetch("http://localhost:5000/login", {
      method: "POST",
      body: new URLSearchParams({
        email: email,
        password: password,
      }),
    });

    const result = await response.json();
    alert(result.message || result.status);

    if (response.ok && result.status === "success") {
      localStorage.setItem("userEmail", email);
      window.location.href = `http://localhost:8501/?email=${encodeURIComponent(email)}`;
    }
  } catch (error) {
    alert("⚠️ Server connection failed — make sure Flask is running on port 5000");
    console.error(error);
  }
});

// UI toggle between signup/login
document.getElementById("switchToLogin").addEventListener("click", (e) => {
  e.preventDefault();
  document.getElementById("signupForm").classList.remove("active");
  document.getElementById("signupForm").classList.add("hidden");
  document.getElementById("loginForm").classList.remove("hidden");
  document.getElementById("loginForm").classList.add("active");
});

document.getElementById("switchToSignup").addEventListener("click", (e) => {
  e.preventDefault();
  document.getElementById("loginForm").classList.remove("active");
  document.getElementById("loginForm").classList.add("hidden");
  document.getElementById("signupForm").classList.remove("hidden");
  document.getElementById("signupForm").classList.add("active");
});


window.addEventListener("load", () => {
  const splash = document.getElementById("splashScreen");
  const bootText = document.getElementById("bootText");
  const text = "LexiAI Initializing...";
  let i = 0;

  // Typing animation effect
  function typeWriter() {
    if (i < text.length) {
      bootText.textContent = text.substring(0, i + 1);
      i++;
      setTimeout(typeWriter, 70); // typing speed
    }
  }

  typeWriter();

  // Remove splash after typing is done
  setTimeout(() => {
    splash.style.opacity = "0";
    splash.style.transition = "opacity 0.5s ease";
    setTimeout(() => {
      splash.style.display = "none";
    }, 600);
  }, 2300); // keep splash visible until text finishes typing
});
