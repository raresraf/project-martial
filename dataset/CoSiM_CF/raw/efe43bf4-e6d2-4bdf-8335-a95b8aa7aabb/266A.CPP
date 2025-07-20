#include <iostream>
using namespace std;
main() {
  int n;
  cin >> n;
  string str;
  cin >> str;
  int total = 0;
  for (int i = 1; i < n; i++) {
    if (str[i] == str[i - 1])
      total++;
  }
  cout << total << endl;
}
