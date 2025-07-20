//
// Created by Taewoo Kim on 3/22/2017.
//

#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(),
            [](unsigned char c) { return std::tolower(c); });
  transform(b.begin(), b.end(), b.begin(),
            [](unsigned char c) { return std::tolower(c); });
  cout << a.compare(b);
  return 0;
}