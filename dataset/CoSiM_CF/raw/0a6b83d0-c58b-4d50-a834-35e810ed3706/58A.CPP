#include <iostream>
#include <string>

using namespace std;

string s;
const string r = "hello";

int main() {
  cin >> s;
  int k = 0;
  for (int i = 0; k < r.length() && i < s.length(); i++)
    if (s[i] == r[k])
      k++;
  cout << (k == r.length() ? "YES" : "NO") << endl;
  {
    int _;
    cin >> _;
  }
}
