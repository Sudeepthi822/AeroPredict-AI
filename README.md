Grama-Urja

Grama-Urja is an MVVM-based Android application built using Kotlin and Jetpack Compose that enables rural communities to monitor power distribution in real time. The app leverages crowdsourced updates and Firebase Cloud Messaging (FCM) alerts to help farmers and residents instantly track grid reliability, power cuts, and restorations.

🚀 Features
⚡ Real-Time Grid Monitoring
Tracks rural power distribution using live cloud database updates.
Synchronizes grid status across active devices in under two seconds.
🔔 Instant FCM Notifications
Sends real-time alerts for:
Power restorations
Power outages
Notifications work even when the app is closed.
📶 Offline Support
Uses Firebase offline disk persistence for local caching.
Allows users to:
View last known power status
Access historical outage logs during connectivity issues.
📊 Intelligent Reliability Analytics
Calculates a localized Zone Reliability Score dynamically.
Uses outage history and recent trends to provide transparency into utility performance.
🛠️ Tech Stack
Technology	Usage
Kotlin	Primary programming language
Jetpack Compose	Modern Android UI toolkit
Material3	UI design system
MVVM Architecture	Application structure
Repository Pattern	Data management layer
Firebase Realtime Database	Real-time cloud database
Firebase Cloud Messaging (FCM)	Push notifications
Kotlin Coroutines	Asynchronous operations
🏗️ Architecture

The application follows the MVVM (Model-View-ViewModel) architecture combined with the Repository Pattern to ensure:

Clean separation of concerns
Scalability
Maintainability
Reactive UI updates
Architecture Flow
UI (Jetpack Compose)
        ↓
ViewModel
        ↓
Repository
        ↓
Firebase Realtime Database
⚙️ Core Functionalities
Reactive Data Flow
Continuously listens to Firebase database changes.
Automatically updates UI with the latest grid information.
Background Notification Services
Uses FCM background services to deliver instant alerts.
Ensures uninterrupted communication with users.
Seamless Offline Experience
Cached data remains accessible without internet connectivity.
Improves reliability in rural network conditions.
Data-Driven Utility Transparency
Analyzes outage patterns and reliability trends.
Helps communities better understand local grid performance.
📱 Target Users
Farmers
Rural households
Local utility monitoring groups
Community volunteers
🔮 Future Enhancements
Predictive outage analysis using Machine Learning
GPS-based outage heatmaps
Community reporting system
Multi-language support
Solar backup integration tracking
