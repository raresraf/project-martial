#include <bits/stdc++.h>
using namespace std;

bool compare(int i, int j) { return i < j; }
int main() {
  string s;

  cin >> s;

  vector<char> inputs;

  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      inputs.push_back(s[i]);
  }

  sort(inputs.begin(), inputs.end(), compare);

  for (int i = 0; i < inputs.size(); i++) {
    if (inputs.size() - 1 == i) {
      cout << inputs[i] << endl;
      break;
    }
    cout << inputs[i] << '+';
  }
  return 0;
}

/**
3+2+1
output:
1+2+3

1+1+3+1+3
output:
1+1+1+3+3

2
output:
2
**/
