
const togglerBtn = document.getElementById('toggler-btn');
const nav = document.getElementById('navigation');

togglerBtn.addEventListener('click', function () {

  if (nav.className === 'nav-off' || nav.className === 'invisible') {
    nav.className = 'nav-on';
  } else {
    nav.className = 'nav-off';
  }

})
