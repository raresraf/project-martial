#include <iostream>
using namespace std;
main() {
  int n;
  cin >> n;
  int x = 0;
  while (n > 0) {
    string word;
    cin >> word;
    int len = word.length();
    if (word[0] == '+' || word[len - 1] == '+')
      x = x + 1;
    else if (word[0] == '-' || word[len - 1] == '-')
      x = x - 1;
    n = n - 1;
  }
  cout << x << endl;
}
