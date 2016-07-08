import Ember from 'ember';

export default Ember.Component.extend({
    websockets: inject.service(),
    socketRef: null,
    images: null,

    init() {
        /*
        * 2) The next step you need to do is to create your actual websocket. Calling socketFor
        * will retrieve a cached websocket if one exists or in this case it
        * will create a new one for us.
        */
        const socket = get(this, 'websockets').socketFor('ws://localhost:8888/reconstruction/images');

        /*
        * 3) The next step is to define your event handlers. All event handlers
        * are added via the `on` method and take 3 arguments: event name, callback
        * function, and the context in which to invoke the callback. All 3 arguments
        * are required.
        */
        socket.on('open', this.openHandler, this);
        socket.on('message', this.messageHandler, this);
        socket.on('close', this.closeHandler, this);

        this.set('socketRef', socket);
    },

    willDestroyElement() {
        const socket = this.get('socketRef');

        /*
        * 4) The final step is to remove all of the listeners you have setup.
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
        this.set('image', message.image);

        console.log(`Message: ${'received image'}`);
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
