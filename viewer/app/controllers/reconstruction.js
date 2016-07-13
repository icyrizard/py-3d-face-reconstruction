import Ember from 'ember';

const { get, inject } = Ember;

export default Ember.Controller.extend({
    title: 'title',
    websockets: inject.service(),
    faces: null,
    image: null,

    image_index: 0,
    n_components: null,
    n_images: null,
    reconstructed: null,

    socketRef: null,

    init() {
        const socket = get(this, 'websockets').socketFor('ws://localhost:8888/reconstruction');

        socket.on('open', this.openHandler, this);
        socket.on('message', this.messageHandler, this);
        socket.on('close', this.closeHandler, this);

        this.set('socketRef', socket);
    },

    willDestroyElement() {
        const socket = this.get('socketRef');

        /*
        * Remove all of the listeners you have setup.
        */
        socket.off('open', this.openHandler);
        socket.off('message', this.messageHandler);
        socket.off('close', this.closeHandler);
    },

    openHandler(event) {
        console.log(`On open event has been called: ${event}`);
    },

    messageHandler(event) {
        var message = JSON.parse(event.data);

        if (message.n_images) {
            this.set('n_images', message.n_images);
        }

        //if (message.image) {
        //    this.set('image', message.image);
        //}

        if (message.reconstructed) {
            this.set('reconstructed', message.reconstructed);
        }

        if (message.error) {
            console.log(message.error);
        }

        //this.get('store').createRecord('face', {
        //    filename: 'Derp',
        //    shape: [1, 2, 3, 4, 5]
        //});
    },

    getReconstruction: Ember.computed('faces', function() {
        console.log(this.get('faces'));
    }),

    closeHandler(event) {
        console.log(`On close event has been called: ${event}`);
    },

    actions: {
        getImage(faceModel) {
            var filename = faceModel.get('filename');
            const socket = this.get('socketRef');

            socket.send(
                JSON.stringify({filename: filename})
            );
        },

        getReconstruction() {
            const socket = this.get('socketRef');

            socket.send(
                JSON.stringify({reconstruction_index: this.get('image_index')}
            ));
        },

        // connects components together
        // handles the upate action passed to a component
        updateComponentConnector(index) {
            this.set('image_index', index);
        }
    }
});
