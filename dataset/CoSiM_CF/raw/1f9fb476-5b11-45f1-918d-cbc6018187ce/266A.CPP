#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  string s;
  cin >> s;

  int count = 0;
  for (int i = 0; i < n - 1; i++)
    if (s.at(i) == s.at(i + 1))
      count++;

  cout << count << endl;

  return 0;
}
