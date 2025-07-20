#include <iostream>
#include <string>
using namespace std;

string toLowerCase(string before) {
  for (int i = 0; i < before.length(); i++) {
    if ((int)before[i] <= 90 && (int)before[i] >= 65) {
      before[i] += 32;
    }
  }

  return before;
}

int main() {

  string one, two;
  cin >> one >> two;

  one = toLowerCase(one);
  two = toLowerCase(two);

  if (one == two)
    cout << 0 << endl;
  else if (one > two)
    cout << 1 << endl;
  else
    cout << -1 << endl;

  return 0;
}
