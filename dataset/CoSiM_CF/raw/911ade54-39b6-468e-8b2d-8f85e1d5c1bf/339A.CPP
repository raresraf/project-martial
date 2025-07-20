// 339A.cpp
// Helpful Maths
// Author: Ikki
// Jun 2, 2015

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  string str;
  cin >> str;
  vector<int> v;
  for (int i = 0; i < str.size(); i++)
    if (str[i] >= '0' && str[i] <= '9')
      v.push_back(str[i] - '0');
  sort(v.begin(), v.end());
  cout << v[0];
  for (int i = 1; i < v.size(); i++)
    cout << "+" << v[i];
  cout << endl;
  return 0;
}