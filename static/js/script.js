document.addEventListener('DOMContentLoaded', function () {
    // Fetch user verification status
    setInterval(fetchVerificationStatus, 1000); // Adjust interval as needed

    function fetchVerificationStatus() {
        const username = new URLSearchParams(window.location.search).get('username');
        fetch(`/user_verified?username=${username}`)
            .then(response => response.json())
            .then(data => {
                console.log('Verification status:', data); // Debugging log
                if (data.verified) {
                    window.location.href = `/user/${data.user}`;
                } else {
                    window.location.href = '/not_verified_user'; // Redirect to the not verified user page
                }
            })
            .catch(error => {
                console.error('Error fetching verification status:', error);
            });
    }
});
