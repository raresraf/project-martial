#include <iostream>
#include <string>

using namespace std;

int main() {
  string testStr;
  int counter = 0;

  cin >> testStr;

  for (int i = 0; i < testStr.length(); i++) {
    if (testStr[i] == '4' || testStr[i] == '7') {
      counter++;
    }
  }

  if (counter == 4 || counter == 7) {
    cout << "YES" << endl;
  }

  else {
    cout << "NO" << endl;
  }

  return 0;
}
