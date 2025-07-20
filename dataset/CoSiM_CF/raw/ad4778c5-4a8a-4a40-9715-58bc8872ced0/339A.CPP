#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  vector<char> num;
  string s;
  cin >> s;

  for (int i = 0; i < s.size(); i++)
    if (s.at(i) != '+')
      num.push_back(s.at(i));

  sort(num.begin(), num.end());

  for (int i = 0; i < num.size() - 1; i++)
    cout << num.at(i) << "+";
  cout << num.at(num.size() - 1) << endl;

  return 0;
}
