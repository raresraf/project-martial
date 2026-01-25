#include <bits/stdc++.h>
using namespace std;

#define vi vector<int>
#define pb push_back

// not working for char 'u'
string resultString(string s) {
  vector<char> arr(s.begin(), s.end());

  for (size_t i = 0; i < arr.size(); i++) {
    switch (arr[i]) {
    case 'a':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'e':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'i':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'o':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'u':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'y':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'A':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'E':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'I':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'O':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'U':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    case 'Y':
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    default:
      continue;
      break;
    }
  }
  vector<char> arr2;
  for (size_t i = 0; i < arr.size(); i++) {
    arr2.pb('.');
    arr2.pb(tolower(arr[i]));
  }
  string p(arr2.begin(), arr2.end());
  return p;
}

int main() {
  string s;
  cin >> s;
  cout << resultString(s) << "\n";
  return 0;
}