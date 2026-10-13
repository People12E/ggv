# Building B.P

Requires a Mac with Xcode 15+ (an IPA cannot be built on Windows or Linux).

    # signed, for your own device / TestFlight
    DEVELOPMENT_TEAM=ABCDE12345 BUNDLE_ID=com.yourname.bp METHOD=development ./scripts/build_ipa.sh   # -> build/ipa/BP.ipa

    # unsigned, to sign afterwards with AltStore / Sideloadly
    UNSIGNED=1 ./scripts/build_ipa.sh                                                                  # -> build/BP-unsigned.ipa

No Mac? Push this folder to a GitHub repo and run the "Build IPA" workflow (Actions tab): it produces the unsigned IPA.

Find your Team ID at developer.apple.com/account (Membership details). A free Apple ID works for installing on your own
device from Xcode, but a distributable / TestFlight IPA needs the paid Developer Program.

Nothing in Sources/ has been compiled yet. Expect to fix a few Swift compile errors on the first build; the build log
will name them. `tools/check_msl.py` and `tools/check_catalog.py` cover the shaders without Xcode.
