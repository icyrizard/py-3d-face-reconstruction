import Ember from 'ember';
import THREE from 'npm:three';
import dat from 'npm:dat-gui';


const ThreeComponent = Ember.Component.extend({
    store: Ember.inject.service(),
    scene: null,

    willRender() {
        if (this.scene) { return; }

        var scene = new THREE.Scene();
        var gui = new dat.GUI();
        var camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 1000
        );

        scene.add(camera);

        var renderer = new THREE.WebGLRenderer();

        // the sidebar 'dat-gui' controls
        var reconstructionControls = {
            index: 0,
            shape_components: 58,
            background_image: true,
        };

        for(var i = 0; i < 15; i++) {
            reconstructionControls['shape_eigen_value_' + i] = 0.0;
        }

        var shapeEigenValueSliders = {};

        this.set('scene', scene);
        this.set('camera', camera);
        this.set('renderer', renderer);
        this.set('gui', gui);
        this.set('controls', reconstructionControls);
        this.set('shapeEigenValueSliders', shapeEigenValueSliders);

        this.get('store').findAll('face').then((faces) => {
            this.set('faces', faces);
            this.addSliders();
        });
    },

    /**
     * Wait for elements to be inserted, else we can not find
     * the threejs container.
     * TODO: try to not use appendChild
     */
    loadGui: Ember.computed('faces', function() {
        //var container = document.getElementById('threejs-container');
        //container.appendChild(this.get('renderer').domElement);

    }),

    /**
     * Adds the 'dat-gui' sliders
     *
     * See:
     * http://learningthreejs.com/blog/2011/08/14/dat-gui-simple-ui-for-demos/
     */
    addSliders() {
        var self = this;
        var gui = this.get('gui');
        var reconstructionControls = this.get('controls');
        var shapeEigenValueSliders = this.get('shapeEigenValueSliders');

        var length = this.get('faces').get('length');

        var index = gui.add(reconstructionControls, 'index', 0, length - 1);
        var shape_components = gui.add(reconstructionControls, 'shape_components', 0, 58);

        var background = gui.add(reconstructionControls, 'background_image');
        var shapeEigenValueControls = gui.addFolder('shape_eigen_values');

        for(var i = 0; i < 15; i++) {
            shapeEigenValueControls.add(reconstructionControls, 'shape_eigen_value_' + i, 0.0, 10.0);
        }

        // on index change
        index.onChange(function(newValue) {
            // update the image_index, which is on the controller
            self.set('image_index', parseInt(newValue));
            self.sendAction('updateIndex', parseInt(newValue));
        });

        background.onChange(function(newValue) {
            self.sendAction('updateBackground', newValue);
        });

        shape_components.onChange(function(newValue) {
            self.sendAction('updateShapeComponents', newValue);
        });

        reconstructionControls.onChange(function(newValue) {
            console.log(newValue);
        });
    }
});

export default ThreeComponent;
