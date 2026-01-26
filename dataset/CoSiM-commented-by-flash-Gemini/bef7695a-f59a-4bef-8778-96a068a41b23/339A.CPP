#include <iostream>
using namespace std;
string s;
int main() {
  cin >> s;
  for (int i = 0; i < s.size(); i += 2)
    for (int j = i + 2; j < s.size(); j += 2)
      if (s[i] > s[j])
        swap(s[i], s[j]);
  cout << s;
}
