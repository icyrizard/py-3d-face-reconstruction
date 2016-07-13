import Ember from 'ember';
import ENV from 'viewer/config/environment';

const ImageLoaderComponent = Ember.Component.extend({
    store: Ember.inject.service(),
    current_face: null,

    imageIndexChanged: Ember.observer('image_index', function() { 
        this.send('updateCurrentFace');
    }),

    loadFaces: Ember.on('didInsertElement', function() {
        this.get('store').findAll('face').then((faces) => {
            this.set('faces', faces);
            this.send('updateCurrentFace');
        });
    }),

    faces: Ember.computed('params.[]', function(){
        return this.get('params')[1];
    }),

    actions: {
        /**
        * Update the current face given the current index
        */
        updateCurrentFace() {
            var face = this.get('faces').objectAt(this.get('image_index'));

            this.set('current_face', face);
            this.set('current_face_filename', ENV.APP.staticURL + face.get('filename'));
        }
    }
});

ImageLoaderComponent.reopenClass({
    positionalParams: ['params']
});

export default ImageLoaderComponent;
