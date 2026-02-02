const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const promptButtons = document.querySelectorAll("[data-prompt]");

const cannedResponses = [
  "Here’s a quick idea: break the task into three tiny steps, and do the first one now.",
  "Try the 2-minute rule: if it takes less than two minutes, do it immediately.",
  "Deep focus works best in 45–90 minute blocks with a short recovery break.",
  "A balanced lunch could be a grain bowl with quinoa, roasted veggies, and a lean protein.",
  "You’ve got this! Small consistent actions build big momentum.",
];

const intentResponses = [
  {
    match: /(productivity|focus|planning|organize)/i,
    message: "Prioritize your top 3 tasks, block time on your calendar, and protect that focus window.",
  },
  {
    match: /(health|lunch|dinner|food)/i,
    message: "Try a protein + fiber combo: grilled chicken, mixed greens, and a side of quinoa.",
  },
  {
    match: /(motivation|encourage|cheer)/i,
    message: "Progress beats perfection. Show up today and you’re already winning.",
  },
  {
    match: /(summary|summarize|benefits)/i,
    message: "Deep focus reduces context switching, improves quality, and helps you finish faster with less stress.",
  },
];

const formatTime = () =>
  new Intl.DateTimeFormat("en", {
    hour: "numeric",
    minute: "numeric",
  }).format(new Date());

const createMessage = (content, role) => {
  const message = document.createElement("article");
  message.classList.add("message", role);

  const bubble = document.createElement("div");
  bubble.classList.add("message-bubble");
  bubble.textContent = content;

  const time = document.createElement("time");
  time.classList.add("message-time");
  time.textContent = formatTime();

  message.appendChild(bubble);
  message.appendChild(time);
  return message;
};

const scrollToBottom = () => {
  chatWindow.scrollTop = chatWindow.scrollHeight;
};

const getBotResponse = (text) => {
  const match = intentResponses.find((item) => item.match.test(text));
  if (match) {
    return match.message;
  }

  return cannedResponses[Math.floor(Math.random() * cannedResponses.length)];
};

const sendMessage = (text) => {
  if (!text.trim()) {
    return;
  }

  const userMessage = createMessage(text, "user");
  chatWindow.appendChild(userMessage);

  const botMessage = createMessage(getBotResponse(text), "bot");
  chatWindow.appendChild(botMessage);

  scrollToBottom();
};

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = messageInput.value;
  sendMessage(message);
  messageInput.value = "";
  messageInput.focus();
});

promptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = button.dataset.prompt;
    messageInput.value = prompt;
    chatForm.requestSubmit();
  });
});
