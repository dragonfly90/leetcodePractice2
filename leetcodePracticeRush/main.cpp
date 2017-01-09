//
//  main.cpp
//  leetcodePracticeRush
//
//  Created by Liang Dong on 7/26/16.
//  Copyright © 2016 Liang Dong. All rights reserved.
//

#include<iostream>
#include<vector>
#include<set>
#include<unordered_map>
#include<unordered_set>
#include<string>
#include<queue>



using namespace std;

class SolutionmaxSumSubmatrix {
    
    int maxs;
    
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        /*
        int m=(int)matrix.size();
        if(m<=0)
            return 0;
        int n=(int)matrix[0].size();
        if(n<=0)
            return 0;
        
        maxs=INT_MIN;
        
        vector<vector<int> > allmatrixsum(m,vector<int>(n,0));
        int area;
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                area=matrix[i][j];
                if(i-1>=0)
                    area+=allmatrixsum[i-1][j];
                if(j-1>=0)
                    area+=allmatrixsum[i][j-1];
                if(i-1>=0&&j-1>=0)
                    area-=allmatrixsum[i-1][j-1];
                
                allmatrixsum[i][j]=area;
            }
        
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                for(int f=i;f<m;f++)
                {
                    for(int l=j;l<n;l++)
                    {
                        area=allmatrixsum[f][l];
                        if(i-1>=0)
                            area-=allmatrixsum[i-1][l];
                        if(j-1>=0)
                            area-=allmatrixsum[f][j-1];
                        if(i-1>=0&&j-1>=0)
                            area+=allmatrixsum[i-1][j-1];
                        if(area<=k)
                            maxs=max(maxs,area);
                    }
                    
                }
                
            }
        }
        
        return maxs;
         */
        if (matrix.empty()) return 0;
        int row = (int)matrix.size();
        int col =(int) matrix[0].size();
        int res = INT_MIN;
        for (int i = 0; i < col; i++) {
            vector<int> sums(row, 0);
            for (int r = i; r < col; r++) {
                for (int k = 0; k < row; k++) {
                    sums[k] += matrix[k][r];
                }
                
                
                set<int> accuSet;
                accuSet.insert(0);
                int curSum = 0, curMax = INT_MIN;
                for (int sum : sums) {
                    curSum += sum;
                    set<int>::iterator it = accuSet.lower_bound(curSum - k);
                    if (it != accuSet.end()) curMax = std::max(curMax, curSum - *it);
                    accuSet.insert(curSum);
                }
                res = std::max(res, curMax);
            }
        }
        return res;
    }
    
};

int best_cumulative_sum(vector<int> ar,int N,int K)
{
    set<int> cumset;
    cumset.insert(0);
    int best=0,cum=0;
    for(int i=0;i<N;i++)
    {
        cum+=ar[i];
        set<int>::iterator sit=cumset.upper_bound(cum-K);
        
        if(sit!=cumset.end())
            best=max(best,cum-*sit);
        
        cumset.insert(cum);
    }
    return best;
}

class SolutionminSubArrayLen {
    public:
        int minSubArrayLen(int s, vector<int>& nums) {
            /*
            sort(nums.begin(),nums.end(),greater<int>());
            int sum=0;
            vector<int> current;
            for(int i=0;i<nums.size();i++)
            {
                sum+=nums[i];
                if(sum>s)
                    return i+1;
                
            }
            
            return 0;
             */
            if(nums.size()<=0)
                return 0;
            int i=0,j=0,sum=0,minv=INT_MAX;
            while(j<nums.size()){
                sum+=nums[j++];
                while(sum>=s){
                    minv=min(minv,j-i);
                    sum-=nums[i++];
                }
            }
            
            return minv==INT_MAX?0:minv;
        }
};

class SolutioncountRangeSum {
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int allcounts=0;
        int nsize=(int)nums.size();
        vector<vector<int> > suma(nsize,vector<int>(nsize,0));
        for(int i=0;i<nsize;i++)
            for(int j=i;j<nsize;j++)
            {
                if(j==i)
                    suma[i][j]=nums[i];
                else
                    suma[i][j]=nums[j]+suma[i][j-1];
                
                if(suma[i][j]>=lower&&suma[i][j]<=upper)
                    allcounts++;
            }
        
        return allcounts;
    }
};

//Definition for singly-linked list.
struct ListNode {
        int val;
        ListNode *next;
        ListNode(int x) : val(x), next(NULL) {}
};


class SolutiongetRandom {
    ListNode* currentHead;
    long long length;
public:
    /** @param head The linked list's head.
     Note that the head is guaranteed to be not null, so it contains at least one node. */
    SolutiongetRandom(ListNode* head) {
        length=0;
        currentHead=head;
        ListNode * temp=head;
        while(temp!=NULL)
        {
            length+=1;
            temp=temp->next;
        }
        
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        ListNode * temp=currentHead;
        
        long long currenl=length;
        while(temp!=NULL)
        {
            if((double)rand()/(RAND_MAX)<1/double(currenl))
                return temp->val;
            else
                currenl-=1;
            temp=temp->next;
        }
        return 0;
        
    }
};


class SolutionShuffle{
    vector<int> shuffleVector;
    int nSize;
public:
    SolutionShuffle(vector<int> nums) {
        shuffleVector.assign(nums.begin(),nums.end());
        nSize=(int)shuffleVector.size();
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return shuffleVector;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> returnvector;
        returnvector.assign(shuffleVector.begin(),shuffleVector.end());
        for(int j = 1; j < nSize; j++) {
            int i = rand()%(j+1);
            int temp=returnvector[i];
            returnvector[i]=returnvector[j];
            returnvector[j]=temp;
        }
        return returnvector;
        
    }
    
};

class SolutionremoveInvalidParentheses {
    
public:
    vector<string> removeInvalidParentheses(string s) {
        vector<string> allstrs;
        set<string> newallstrs;
        vector<string> tempstrs;
        set<string> strset;
        allstrs.clear();
        allstrs.push_back(s);
        
        while(1)
        {
            strset.clear();
            bool valid=true;
            for(auto tempstr:allstrs)
            {
                if(isvalid(tempstr))
                {
                    strset.insert(tempstr);
                }
                else
                    valid=false;
            }
            
            
            if(!strset.empty())
            {
                return vector<string>(strset.begin(),strset.end());
            }
            
            newallstrs.clear();
            for(auto tempstr:allstrs)
            {
                for(int i=1;i<tempstr.size();i++)
                    newallstrs.insert(tempstr.substr(0,i)+tempstr.substr(i+1));
                newallstrs.insert(tempstr.substr(1));
            }
            //for(auto i:newallstrs)
            //    cout<<i<<endl;
            allstrs.clear();
            allstrs.assign(newallstrs.begin(),newallstrs.end());
            
        }
    }
    
    bool isvalid(string s){
        int ctr=0;
        for(auto i:s)
        {
            if(i=='(')
                ctr+=1;
            else
                if(i==')')
                    ctr-=1;
            if(ctr<0)
                return false;
        }
        
        return ctr==0;
    }
    

    
};

class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        unordered_map<int,int> allset;
        
        for(int i=0;i<numbers.size();i++)
        {
            
            if(allset.find(numbers[i])!=allset.end())
            {
                vector<int> returnIndex;
                returnIndex.push_back(allset[numbers[i]]);
                returnIndex.push_back(i);
                return returnIndex;
            }
            else
                allset[target-numbers[i]]=i;
        }
        return vector<int>();
    }
    
    vector<int> twoSum2(vector<int>& numbers, int target){
        int lo=0, hi=(int)numbers.size()-1;
        while (numbers[lo]+numbers[hi]!=target){
            if (numbers[lo]+numbers[hi]<target){
                lo++;
            } else {
                hi--;
            }
        }
        return vector<int>({lo+1,hi+1});
    }
};

class SolutioncanConstruct {
public:
    bool canConstruct(string ransomNote, string magazine) {
        int magazine_nums[26];
        for(int i=0;i<26;i++)
        {
            magazine_nums[i]=0;
        }
        
        for(auto cm:magazine)
        {
            magazine_nums[int(cm-'a')]++;
        }
        
        for(auto cm:ransomNote)
        {
            magazine_nums[int(cm-'a')]-=1;
            if(magazine_nums[int(cm-'a')]<0)
                return false;
            
        }
       
        return true;
    }
};


class SolutionisSubsequence {
public:
    bool isSubsequence(string s, string t) {
        
        int i=0;
        int j=0;
        while(i<s.size()&&j<t.size())
        {
            if(s[i]==t[j])
            {
                i++;
                j++;
            }
            else
                j++;
        }
        
        if(i==s.size())
            return true;
        else
            return false;
    }
};

// Definition for a binary tree node.
 struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  };

class SolutionsumOfLeftLeaves {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        int sum = 0;
        if(!root) return sum;
        preOrder(root,sum);
        return sum;
        
    }
    void preOrder(TreeNode* root,int& sum)
    {
        if(root) {
            if(root->left != NULL && root->left->left == NULL && root->left->right == NULL)
                sum += root->left->val;
            preOrder(root->left, sum);
            preOrder(root->right, sum);
        }
    }
};


class SolutionreconstructQueue {
public:
    vector<pair<int, int>> reconstructQueue(vector<pair<int, int>>& people) {
        auto compa = [](const pair<int, int>& p1, const pair<int, int>& p2)
        { return p1.first > p2.first || (p1.first == p2.first && p1.second < p2.second); };
        sort(people.begin(), people.end(), compa);
        vector<pair<int, int>> res1;
        for (auto& p : people)
            res1.insert(res1.begin() + p.second, p);
        return res1;
    }
};

class SolutionaddStrings {
public:
    string addStrings(string num1, string num2) {
        int i;
        std::reverse(num1.begin(), num1.end());
        std::reverse(num2.begin(), num2.end());
        int len1 = (int)num1.size();
        int len2 = (int)num2.size();
        int minlen;
        
        if(len1>len2)
            minlen = len2;
        else
            minlen = len1;
        
        string sumstr;
        int temp=0;
        int add=0;
      
        for(i=0;i<minlen;i++)
        {
            temp=int(num1[i]-'0')+int(num2[i]-'0')+add;
            cout<<temp<<endl;
            if(temp>=10)
            {
                temp=temp-10;
                add=1;
            }
            else
                add=0;
            sumstr.push_back(char(temp+'0'));
        }
        while(len1>i)
        {
            temp=int(num1[i]-'0')+add;
            if(temp>=10)
            {
                temp=temp-10;
                add=1;
            }
            else
                add=0;
            sumstr.push_back(char(temp+'0'));
            i++;
        }
        while(len2>i)
        {
            temp=int(num2[i]-'0')+add;
            if(temp>=10)
            {
                temp=temp-10;
                add=1;
            }
            else
                add=0;
            sumstr.push_back(char(temp+'0'));
            i++;
        }
        
        if(add!=0)
            sumstr.push_back(char(add+'0'));
        
        std::reverse(sumstr.begin(),sumstr.end());
        return sumstr;
    }
};

/*
 def readBinaryWatch(self, num):
 return ['%d:%02d' % (h, m)
 for h in range(12) for m in range(60)
 if (bin(h) + bin(m)).count('1') == num]
 */


class SolutionreadBinaryWatch {
public:
    string changetotime(int i, int j){
        string str;
        if(j<10)
            str = to_string(i)+":0"+to_string(j);
        else
            str = to_string(i)+":"+to_string(j);
        return str;
    }
    
    int numofdigits(int m)
    {
        int sumd=0;
        while(m>0)
        {
            sumd=sumd+(m&1);
            m=m>>1;
        }
        return sumd;
    }
    vector<string> readBinaryWatch(int num) {
        vector<string> allstrs;
        
        for(int i=0;i<12;i++)
            for(int j=0;j<60;j++)
            {
                if(numofdigits(i)+numofdigits(j)==num)
                {
                    allstrs.push_back(changetotime(i, j));
                }
            }
        
        return allstrs;
    }
};

class SolutiontoHex {
public:
    string toHex(int num) {
        unsigned long positive;
        unsigned long large;
        large=1;
        large=large<<32;
        if(num<0)
        {
            positive=large+num;
        }
        else
            positive=num;
        
        cout<<positive<<endl;
        
        if(positive==0)
            return "0";
        
        string allstrs;
        int temp;
        
        while(positive>0)
        {
            temp=positive%16;
            positive/=16;
            
            if(temp<10)
                allstrs.push_back(char('0'+temp));
            else
            {
                temp-=10;
                allstrs.push_back(char('a'+temp));
            }
        }
        
        reverse(allstrs.begin(), allstrs.end());
        return allstrs;
        
    }
};

class SolutionkthSmallest {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = (int)matrix.size();
        int le = matrix[0][0], ri = matrix[n-1][n-1];
        int mid = 0;
        while(le < ri)
        {
            mid = le + (ri-le)/2;
            int num = 0;
            for(int i=0; i<n; i++)
            {
                int pos = (int)(upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin());
                num +=pos;
            }
            if(num<k)
            {
                le = mid +1;
            }
            else
            {
                ri = mid;
            }
        }
        
        return le;
    }
};

class SolutionkthSmallest2 {
public:
    struct compare
    {
        bool operator()(const pair<int,pair<int, int> >& a, const pair<int,pair<int, int> >& b)
        {
            return a.first>b.first;
        }
    };
    int kthSmallest(vector<vector<int>>& arr, int k) {
        
        int n=(int)arr.size();
        int m=(int)arr[0].size();
        
        std::priority_queue< pair<int,pair<int, int> >, vector<pair<int, pair<int, int> > >, compare > p;
        
        for(int i=0;i<n;i++)
            p.push(make_pair(arr[i][0],make_pair(i,0)));
        
        int x=k;
        int ans=k;
        while(x--)
        {
            int e=p.top().first;
            int i=p.top().second.first;
            int j=p.top().second.second;
            ans=e;
            p.pop();
            if(j!=m-1)
            p.push(make_pair(arr[i][j+1],make_pair(i,j+1)));
        }
        return ans;
        
    }
};

template<typename T> void print_queue(T& q) {
    while(!q.empty()) {
        std::cout << q.top() << " ";
        q.pop();
    }
    std::cout << '\n';
}

class SolutionfizzBuzz {
public:
    vector<string> fizzBuzz(int n) {
        vector<string> alls;
        for(int i=1;i<=n;i++)
        {
            if(i%15==0)
            {
                alls.push_back("FizzBuzz");
            }
            else
            {
                if(i%5==0)
                    alls.push_back("Buzz");
                else
                {
                    if(i%3==0)
                        alls.push_back("Fizz");
                    else
                    {
                        
                        alls.push_back(std::to_string(i));
                    }
                }
            }
        }
        
        return alls;
    }
};


class SolutionnumberOfArithmeticSlices {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        int n=(int)A.size();
        vector<int> dp(n,0);
        
        if(n<3) return 0;
        
        if(A[2]-A[1]==A[1]-A[0])
            dp[2]=1;
        int result = dp[2];
        for(int i=3;i<n;i++){
            if(A[i]-A[i-1]==A[i-1]-A[i-2])
                dp[i]=dp[i-1]+1;
            result+=dp[i];
        }
        return result;
    }
};

class SolutioncountBattleships {
public:
    int countBattleships(vector<vector<char>>& board) {
        int count = 0;
        for(int i=0;i<board.size();i++)
        for(int j=0;j<board[0].size();j++)
        if(board[i][j]=='X' && (i==0 || board[i-1][j]!='X') && (j==0 || board[i][j-1]!='X')) count++;
        return count;
    }
};

class SolutionhammingDistance {
public:
    int hammingDistance(int x, int y) {
        int z=x^y;
        int sumn=0;
        while(z!=0)
        {
            sumn+=z&1;
            z=z>>1;
        }
        return sumn;
    }
};

/*
 public class Solution {
	Map<String,List<String>> map;
	List<List<String>> results;
 public List<List<String>> findLadders(String start, String end, Set<String> dict) {
 results= new ArrayList<List<String>>();
 if (dict.size() == 0)
 return results;
 
 int min=Integer.MAX_VALUE;
 
 Queue<String> queue= new ArrayDeque<String>();
 queue.add(start);
 
 map = new HashMap<String,List<String>>();
 
 Map<String,Integer> ladder = new HashMap<String,Integer>();
 for (String string:dict)
 ladder.put(string, Integer.MAX_VALUE);
 ladder.put(start, 0);
 
 dict.add(end);
 //BFS: Dijisktra search
 while (!queue.isEmpty()) {
 
 String word = queue.poll();
 
 int step = ladder.get(word)+1;//'step' indicates how many steps are needed to travel to one word.
 
 if (step>min) break;
 
 for (int i = 0; i < word.length(); i++){
 StringBuilder builder = new StringBuilder(word);
 for (char ch='a';  ch <= 'z'; ch++){
 builder.setCharAt(i,ch);
 String new_word=builder.toString();
 if (ladder.containsKey(new_word)) {
 
 if (step>ladder.get(new_word))//Check if it is the shortest path to one word.
 continue;
 else if (step<ladder.get(new_word)){
 queue.add(new_word);
 ladder.put(new_word, step);
 }else;// It is a KEY line. If one word already appeared in one ladder,
 // Do not insert the same word inside the queue twice. Otherwise it gets TLE.
 
 if (map.containsKey(new_word)) //Build adjacent Graph
 map.get(new_word).add(word);
 else{
 List<String> list= new LinkedList<String>();
 list.add(word);
 map.put(new_word,list);
 //It is possible to write three lines in one:
 //map.put(new_word,new LinkedList<String>(Arrays.asList(new String[]{word})));
 //Which one is better?
 }
 
 if (new_word.equals(end))
 min=step;
 
 }//End if dict contains new_word
 }//End:Iteration from 'a' to 'z'
 }//End:Iteration from the first to the last
 }//End While
 
 //BackTracking
 LinkedList<String> result = new LinkedList<String>();
 backTrace(end,start,result);
 
 return results;
 }
 private void backTrace(String word,String start,List<String> list){
 if (word.equals(start)){
 list.add(0,start);
 results.add(new ArrayList<String>(list));
 list.remove(0);
 return;
 }
 list.add(0,word);
 if (map.get(word)!=null)
 for (String s:map.get(word))
 backTrace(s,start,list);
 list.remove(0);
 }
 }
 */

class SolutionfindLadders {
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, unordered_set<string> &dict) {
        vector<vector<string> > paths;
        vector<string> path(1, beginWord);
        if(beginWord==endWord)
        {
            paths.push_back(path);
            return paths;
        }
        unordered_set<string> forward, backward;
        forward.insert(beginWord);
        backward.insert(endWord);
        unordered_map<string, vector<string> > tree;
        bool reversed = false;
        if(buildTree(forward,backward,dict,tree,reversed))
            getPath(beginWord, endWord, tree, path, paths);
        return paths;
    }
private:
    bool buildTree(unordered_set<string> &forward, unordered_set<string> &backward, unordered_set<string> &dict, unordered_map<string, vector<string> > &tree, bool reversed)
    {
        if (forward.empty()) return false;
        if (forward.size() > backward.size())
        return buildTree(backward, forward, dict, tree, !reversed);
        for (auto &word: forward) dict.erase(word);
        for (auto &word: backward) dict.erase(word);
        unordered_set<string> nextLevel;
        bool done = false; //in case of invalid further searching;
        for (auto &it: forward) //traverse each word in the forward -> the current level of the tree;
        {
            string word = it;
            for (auto &c: word)
            {
                char c0 = c; //store the original;
                for (c = 'a'; c <= 'z'; ++c) //try each case;
                {
                    if (c != c0) //avoid futile checking;
                    {
                        if (backward.count(word))  //using count is an accelerating method;
                        {
                            done = true;
                            !reversed ? tree[it].push_back(word) : tree[word].push_back(it); //keep the tree generation direction consistent;
                        }
                        else if (!done && dict.count(word))
                        {
                            nextLevel.insert(word);
                            !reversed ? tree[it].push_back(word) : tree[word].push_back(it);
                        }
                    }
                }
                c = c0; //restore the word;
            }
        }
        return done || buildTree(nextLevel, backward, dict, tree, reversed);
    }
    
    void getPath(string &beginWord, string &endWord, unordered_map<string, vector<string> > &tree, vector<string> &path, vector<vector<string> > &paths) //using reference can accelerate;
    {
        if (beginWord == endWord) paths.push_back(path); //till the end;
        else
        {
            for (auto &it: tree[beginWord])
            {
                path.push_back(it);
                getPath(it, endWord, tree, path, paths); //DFS retrieving the path;
                path.pop_back();
            }
        }
    }
};

class SolutionminMoves2 {
public:
    int minMoves2(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        //Arrays.sort(nums);
        int i = 0, j = (int)nums.size()-1;
        int count = 0;
        while(i < j){
            count += nums[j]-nums[i];
            i++;
            j--;
        }
        return count;
    }
};

/*
 Arrays.sort(g);
 Arrays.sort(s);
 int i = 0;
 for(int j=0;i<g.length && j<s.length;j++) {
	if(g[i]<=s[j]) i++;
 }
 return i;
 */

class SolutionfindContentChildren {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(),g.end());
        sort(s.begin(),s.end());
        int i = 0;
        for(int j=0;i<g.size()&&j<s.size();j++){
            if(g[i]<=s[j])
                i++;
        }
        
        return i;
                    
    }
};


class SolutionfindDisappearedNumbers {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        vector<int> ret;
        for(int i=0;i<nums.size();i++){
            int val = abs(nums[i])-1;
            if(nums[val]>0){
                nums[val] = -nums[val];
            }
        }
        
        for(int i=0;i<nums.size();i++){
            if(nums[i]>0){
                ret.push_back(i+1);
            }
        }
        
        return ret;
    }
};

class SolutiontotalHammingDistance {
public:
    int countn(int a){
        int sumn=0;
        while(a>0)
        {
            sumn+=a&1;
            a>>=1;
        }
        return sumn;
    }
    int totalHammingDistance(vector<int>& nums) {
        int total = 0, n = (int)nums.size();
        for (int j=0;j<32;j++) {
            int bitCount = 0;
            for (int i=0;i<n;i++)
            bitCount += (nums[i] >> j) & 1;
            total += bitCount*(n - bitCount);
        }
        return total;
    }
};


class SolutionminMoves {
public:
    unsigned long sumn(vector<int>& nums){
        unsigned long sumn=0;
        for(auto i:nums)
        {
            sumn+=i;
        }
        return sumn;
    }
    
    int minMoves(vector<int>& nums) {
        return (int)(sumn(nums) - nums.size() * *min_element(begin(nums), end(nums)));
    }
};

int main(){
    SolutionfizzBuzz solution;
    vector<string> alls=solution.fizzBuzz(15);
    for(auto i:alls)
        cout<<i<<endl;
    /*
    std::priority_queue<int> q;
    vector<int> all;
    all.push_back(1);
    all.push_back(8);
    all.push_back(5);
    
    for(int n: all)
        q.push(n);
    
    print_queue(q);
    
    std::priority_queue<int, std::vector<int>, std::greater<int> > q2;
    
    for(int n:all)
        q2.push(n);
    
    print_queue(q2);
    
    // Using lambda to compare elements.
    auto cmp = [](int left, int right) { return (left ^ 1) < (right ^ 1);};
    std::priority_queue<int, std::vector<int>, decltype(cmp)> q3(cmp);
    
    for(int n : {1,8,5,6,3,4,0,9,7,2})
        q3.push(n);
    
    print_queue(q3);
     */
    /*
    vector<int> allnums;
    allnums.push_back(1);
    allnums.push_back(2);
    sort(allnums.begin(),allnums.end());
    for(auto a: allnums)
        cout<<a<<endl;
     */
    
    //SolutiontoHex solution;
    //cout<<solution.toHex(-1)<<endl;
    /*
     SolutionreadBinaryWatch solution;
     vector<string> allstrs=solution.readBinaryWatch(0);
     
     for(auto i:allstrs)
     cout<<i<<endl;
     */
    //SolutioncanConstruct solution;
    //cout<<solution.canConstruct("a", "b")<<endl;
    /*
     SolutionmaxSumSubmatrix solution;
     vector<vector<int> > testmatrix;
     int k=2;
     int matrix1[]={1,0,1};
     int matrix2[]={0,-2,3};
     testmatrix.resize(2);
     testmatrix[0].assign(matrix1,matrix1+2);
     testmatrix[1].assign(matrix2,matrix2+2);
     
     cout<<solution.maxSumSubmatrix(testmatrix, k)<<endl;
     */
    /*
     SolutionremoveInvalidParentheses mysolution;
     vector<string> allstrs=mysolution.removeInvalidParentheses("()())()");
     for(auto str: allstrs)
     cout<<str<<endl;
     */
}

class SolutionislandPerimeter {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int m=(int)grid.size();
        int n=0;
        int sum=0;
        if(m>0)
            n=(int)grid[0].size();
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
        {
            sum=sum+search(grid,i,j,m,n);
        }
        return sum;
    }
    
    int search(vector<vector<int> >& grid, int i, int j, const int& m, const int& n)
    {
        if(grid[i][j]==0)
            return 0;
        else
        {
            int current=4;
            if(i>=1&&grid[i-1][j]==1)
                current-=1;
            if(j>=1&&grid[i][j-1]==1)
                current-=1;
            if(i<m-1&&grid[i+1][j]==1)
                current-=1;
            if(j<n-1&&grid[i][j+1]==1)
                current-=1;
            return current;
        }
    }
};

/*
 int sumOfLeftLeaves(TreeNode* root) {
 
 if(root==NULL)
 return 0;
 return iterateleft(root);
 
 }
 int iterateleft(TreeNode* root)
 {
 if(root==NULL)
 return 0;
 
 return iterateleft2(root->left)+iterateleft(root->right);
 }
 int iterateleft2(TreeNode* root)
 {
 if(root==NULL)
 return 0;
 
 if(root->left==NULL&&root->right==NULL)
 {
 return root->val;
 }
 else
 {
 return iterateleft2(root->left)+iterateleft(root->right);
 }
 }
 */


    
