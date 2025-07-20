#include <iostream>
#include <string>

using namespace std;

int main() {
  string input;
  getline(cin, input);

  int n;
  n = input.length();

  int flag = 0;

  // cout<<n<<'\n';

  for (int i = 0; i < n; i++) {
    if ((flag == 0 && input.at(i) == 'h') ||
        (flag == 1 && input.at(i) == 'e') ||
        ((flag == 2 || flag == 3) && input.at(i) == 'l') ||
        (flag == 4 && input.at(i) == 'o')) {
      flag++;
    }
    if (flag == 5) {
      cout << "YES";
      return 0;
    }
    // cout<<i;
  }

  // cout<<flag;

  cout << "NO";
  return 0;
}
