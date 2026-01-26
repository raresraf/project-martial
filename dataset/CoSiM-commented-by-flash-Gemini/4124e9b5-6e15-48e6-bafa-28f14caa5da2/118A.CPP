#include <bits/stdc++.h>

using namespace std;

int main() {

  string input;
  cin >> input;

  for (int i = 0; i < input.size(); ++i) {
    char elementAti = tolower(input[i]);
    if (elementAti == 'a' || elementAti == 'e' || elementAti == 'i' ||
        elementAti == 'o' || elementAti == 'u' || elementAti == 'y')
      continue;
    else
      cout << "." << elementAti;
  }

  cout << endl;
}