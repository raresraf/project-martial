#include <iostream>

using namespace std;

int main() {
  string word;
  string prev;
  auto count = 0;

  cin >> word;

  for (auto c : word) {
    if (prev.find(c) == string::npos) {
      count++;
    }
    prev += c;
  }

  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}
