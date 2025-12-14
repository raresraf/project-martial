#include <iostream>
#include <string>

using namespace std;

int main() {
  int counter = 0;
  int inc = 0;
  char hello[5] = {'h', 'e', 'l', 'l', 'o'};
  string input;
  cin >> input;
  for (int i = 0; i < 5; i++) {
    for (inc; inc < input.length(); inc++) {
      if (hello[i] == input.at(inc)) {
        counter++;
        inc++;
        break;
      }
    }
  }
  if (counter == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
