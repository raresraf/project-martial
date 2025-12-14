#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  string words[100];

  for (int i = 0; i < n; i++)
    cin >> words[i];

  for (int i = 0; i < n; i++) {
    int wordLength = words[i].size();
    if (wordLength > 10)
      cout << words[i][0] << wordLength - 2 << words[i][wordLength - 1] << endl;
    else
      cout << words[i] << endl;
  };
  return 0;
}