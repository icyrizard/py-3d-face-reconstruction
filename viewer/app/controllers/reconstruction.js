import Ember from 'ember';

const { get, inject } = Ember;

export default Ember.Controller.extend({
    websockets: inject.service(),

    title: 'title',
    faces: null,
    image: null,

    image_index: 0,
    background_image: true,
    shape_components: null,
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

        console.log('Websockets: Removed all handlers');
    },

    openHandler(event) {
        console.log(`On open event has been called: ${event}`);

        // get the reconstruction right after the socket opened
        this.send('getReconstruction');
    },

    messageHandler(event) {
        var message = JSON.parse(event.data);

        if (message.n_images) {
            this.set('n_images', message.n_images);
        }

        if (message.reconstructed) {
            this.set('reconstructed', message.reconstructed);
        }

        if (message.error) {
            console.log(message.error);
        }

        this.set('loading', false);
    },

    getReconstruction: Ember.observer(
            'image_index', 'background_image', 'shape_components', function() {
        this.send('getReconstruction');
    }),

    closeHandler(event) {
        console.log(`On close event has been called: ${event}`);
    },

    actions: {
        getImage(faceModel) {
            this.set('loading', true);
            var filename = faceModel.get('filename');
            const socket = this.get('socketRef');

            socket.send(
                JSON.stringify({filename: filename})
            );
        },

        getReconstruction() {
            this.set('loading', true);
            const socket = this.get('socketRef');

            socket.send(
                JSON.stringify({
                    reconstruction: {
                        reconstruction_index: this.get('image_index'),
                        background_image: this.get('background_image'),
                        shape_components: this.get('shape_components')
                    }

                }
            ));
        },

        // connects components together
        // handles the upate action passed to a component
        updateIndexComponentConnector(index) {
            this.set('image_index', index);
        },

        updateBackgroundComponentConnector(showBackground) {
            this.set('background_image', showBackground);
        },

        updateShapeComponents(components) {
            console.log('shape_components', components);
            this.set('shape_components', components);
        }
    }
});
