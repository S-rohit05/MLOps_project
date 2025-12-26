document.addEventListener('DOMContentLoaded', () => {
    // Inputs
    const inputs = {
        credit_score: document.getElementById('credit_score'),
        age: document.getElementById('age'),
        tenure: document.getElementById('tenure'),
        balance: document.getElementById('balance'),
        products_number: document.getElementById('products_number'),
        estimated_salary: document.getElementById('estimated_salary'),
        credit_card: document.getElementById('credit_card'),
        active_member: document.getElementById('active_member')
    };

    // Displays
    const displays = {
        credit_score: document.getElementById('disp_credit_score'),
        age: document.getElementById('disp_age'),
        tenure: document.getElementById('disp_tenure'),
        balance: document.getElementById('disp_balance'),
        products_number: document.getElementById('disp_products'),
        estimated_salary: document.getElementById('disp_salary')
    };

    // UI Elements
    const gaugeFill = document.getElementById('gauge-fill');
    const probText = document.getElementById('prob-text');
    const riskLabel = document.getElementById('risk-label');
    const factorList = document.getElementById('factor-list');
    const predictBtn = document.getElementById('predict-btn');

    // --- interactivity ---

    // Update numeric displays on slider input
    Object.keys(displays).forEach(key => {
        inputs[key].addEventListener('input', (e) => {
            let val = e.target.value;
            // Format currency
            if (key === 'balance' || key === 'estimated_salary') {
                val = '$' + parseInt(val).toLocaleString();
            }
            displays[key].textContent = val;

            // Auto-trigger prediction (debounce could be added)
            // predict(); 
        });
    });

    // Handle Toggles
    [inputs.credit_card, inputs.active_member].forEach(btn => {
        btn.addEventListener('click', () => {
            const current = btn.dataset.value === '1';
            const newVal = current ? '0' : '1';
            btn.dataset.value = newVal;
            btn.classList.toggle('active', newVal === '1');
            btn.textContent = (btn.id === 'credit_card' ? 'Credit Card: ' : 'Active Member: ') + (newVal === '1' ? 'YES' : 'NO');
        });
    });

    async function predict() {
        // Collect Data
        const payload = {
            credit_score: parseInt(inputs.credit_score.value),
            age: parseInt(inputs.age.value),
            tenure: parseInt(inputs.tenure.value),
            balance: parseFloat(inputs.balance.value),
            products_number: parseInt(inputs.products_number.value),
            estimated_salary: parseFloat(inputs.estimated_salary.value),
            credit_card: parseInt(inputs.credit_card.dataset.value),
            active_member: parseInt(inputs.active_member.dataset.value)
        };

        predictBtn.textContent = "Analyzing...";

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            updateDashboard(data);
        } catch (error) {
            console.error(error);
            alert("Error connecting to model API");
        } finally {
            predictBtn.textContent = "Update Risk Analysis";
        }
    }

    function updateDashboard(data) {
        const prob = data.churn_probability; // 0.0 to 1.0
        const percentage = (prob * 100).toFixed(1) + '%';

        // 1. Update Gauge Rotation (-135deg to +45deg = 180deg range)
        // 0% -> -135deg, 100% -> 45deg
        const rotation = -135 + (prob * 180);
        gaugeFill.style.transform = `rotate(${rotation}deg)`;

        // 2. Color Scaling
        let color = '#22c55e'; // Green
        let riskText = "Low Risk";

        if (prob > 0.4 && prob <= 0.7) {
            color = '#f59e0b'; // Orange
            riskText = "Medium Risk";
        } else if (prob > 0.7) {
            color = '#ef4444'; // Red
            riskText = "High Risk";
        }

        gaugeFill.style.borderColor = color;
        probText.style.color = color;
        riskLabel.style.color = color;

        // 3. Text Updates
        probText.textContent = percentage;
        riskLabel.textContent = riskText;

        // 4. Update Factors
        factorList.innerHTML = '';
        if (data.factors && data.factors.length > 0) {
            data.factors.forEach(factor => {
                const chip = document.createElement('div');
                chip.className = 'factor-chip';
                chip.textContent = factor;
                factorList.appendChild(chip);
            });
        } else {
            const chip = document.createElement('div');
            chip.className = 'factor-chip';
            chip.textContent = "Standard Profile";
            chip.style.opacity = 0.5;
            factorList.appendChild(chip);
        }
    }

    // Attach listener
    predictBtn.addEventListener('click', predict);

    // Initial run
    predict();
});
