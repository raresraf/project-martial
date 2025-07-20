#include <iostream>
using namespace std;

int main() {
  string input;
  cin >> input;
  int flag = 1;

  for (int i = 0; i < input.length() - 1; i++) {
    if (input[i] == input[i + 1]) {
      ++flag;
      if (flag == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else
      flag = 1;
  }
  cout << "NO" << endl;

  return 0;
}