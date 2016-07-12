import Ember from 'ember';
import THREE from 'npm:three';
import dat from 'npm:dat-gui';

export default Ember.Component.extend({
    store: Ember.inject.service(),

    willRender() {
        if (this.scene) {
            return;
        }

        var scene = new THREE.Scene();
        var gui = new dat.GUI();

        var camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 1000
        );

        scene.add(camera);

        var renderer = new THREE.WebGLRenderer();

        this.set('scene', scene);
        this.set('camera', camera);
        this.set('renderer', renderer);
        this.set('gui', gui);
    },

    /**
     * Wait for elements to be inserted, else we can not find
     * the threejs container.
     * TODO: try to not use appendChild
     */
    loadGui: Ember.on('didInsertElement', function() {
        //var container = document.getElementById('threejs-container');
        //container.appendChild(this.get('renderer').domElement);

        this.addSliders();
    }),

    addSliders() {
        var self = this;
        var gui = this.get('gui');

        var obj = {
            name: "Image filename",
            index: 0
        };

        var components = {
            name: "Components",
            components: 0
        };

        var imagesSlider = gui.add(obj, "index").min(0).max(
                this.n_images - 1).step(1);

        gui.add(components, "components").min(0).max(this.n_images - 1).step(1);

        imagesSlider.onChange(function(newValue) {
            self.set('image_index', newValue);
        });
    }
});
