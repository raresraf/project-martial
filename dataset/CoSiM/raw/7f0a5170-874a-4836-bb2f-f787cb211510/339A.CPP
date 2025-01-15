#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  string s, result = "";
  vector<char> v;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= '0' && s[i] <= '9')
      v.push_back(s[i]);
  }
  sort(v.begin(), v.end());
  result += v[0];
  for (int i = 1; i < v.size(); i++) {
    result += '+';
    result += v[i];
  }
  cout << result << endl;
  return 0;
}
