#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string first, second;
  cin >> first >> second;

  transform(first.begin(), first.end(), first.begin(), (int (*)(int))tolower);
  transform(second.begin(), second.end(), second.begin(),
            (int (*)(int))tolower);

  if (first > second) {
    cout << "1";
  } else if (first < second) {
    cout << "-1";
  } else
    cout << "0";
  return 0;
}
