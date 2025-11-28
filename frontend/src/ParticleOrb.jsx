import React, { useEffect, useRef } from 'react';

const ParticleOrb = ({ isActive }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let particles = [];
    
    const resize = () => {
      canvas.width = 300;
      canvas.height = 300;
    };
    resize();

    class Particle {
      constructor() {
        this.angle = Math.random() * Math.PI * 2;
        this.radius = 50 + Math.random() * 40;
        this.speed = 0.02 + Math.random() * 0.03;
        this.size = 1 + Math.random() * 2;
        this.z = Math.random() * Math.PI * 2;
      }

      update() {
        this.angle += this.speed * (isActive ? 2 : 0.5);
        this.z += 0.01 * (isActive ? 2 : 0.5);
        
        // 3D-ish projection
        this.x = canvas.width / 2 + Math.cos(this.angle) * this.radius * Math.sin(this.z);
        this.y = canvas.height / 2 + Math.sin(this.angle) * this.radius;
        this.alpha = 0.5 + 0.5 * Math.cos(this.z);
      }

      draw() {
        ctx.fillStyle = `rgba(6, 182, 212, ${this.alpha})`; // Cyan
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (let i = 0; i < 100; i++) {
      particles.push(new Particle());
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      particles.forEach(p => {
        p.update();
        p.draw();
      });

      // Draw Core
      if (isActive) {
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, 40, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.1)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isActive]);

  return <canvas ref={canvasRef} className="w-[300px] h-[300px]" />;
};

export default ParticleOrb;
