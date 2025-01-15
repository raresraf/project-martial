#include <iostream>
using namespace std;

int main() {

  int n;
  int counter = 0;
  cin >> n;
  string str;
  cin >> str;

  for (int i = 0; i < n - 1; i++) {

    if (str[i] == str[i + 1]) {
      counter++;
    }
  }

  cout << counter << endl;

  return 0;
}
