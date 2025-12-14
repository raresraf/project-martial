#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {

  int n;
  cin >> n;
  vector<string> words(n);
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }

  for (int i = 0; i < n; i++) {
    if (words[i].size() > 10) {
      cout << words[i].at(0) << words[i].size() - 2
           << words[i].at(words[i].size() - 1) << endl;
    } else {
      cout << words[i] << endl;
    }
  }
}