import Ember from 'ember';

const { get, inject } = Ember;

export default Ember.Controller.extend({
    title: 'title',
    websockets: inject.service(),
    socketRef: null,
    image: null,
    n_images: 0,

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
        //var message = JSON.parse(event.data);
        //this.set('n_images', message.n_images);
        console.log(event);
        console.log(event.data.n_images);
        console.log(`On open event has been called: ${event}`);
    },

    messageHandler(event) {
        //var message = JSON.parse(event.data);
        //this.set('image', message.image);

        //console.log(`Message: ${'received image'}`);
    },

    closeHandler(event) {
        console.log(`On close event has been called: ${event}`);
    },

    actions: {
        get_image() {
            const socket = this.get('socketRef');
            socket.send(
                JSON.stringify({image_index: 1}
            ));
        },
    }
});
