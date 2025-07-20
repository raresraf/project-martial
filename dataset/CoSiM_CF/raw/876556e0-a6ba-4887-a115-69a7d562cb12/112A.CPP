#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  string first, second;
  cin >> first >> second;
  transform(first.begin(), first.end(), first.begin(), ::tolower);
  transform(second.begin(), second.end(), second.begin(), ::tolower);
  if (first < second)
    cout << -1;
  else if (first > second)
    cout << 1;
  else
    cout << 0;
  return 0;
}
