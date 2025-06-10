import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.Map.Entry;

import org.jgrapht.ext.ComponentNameProvider;
import org.jgrapht.ext.DOTExporter;
import org.jgrapht.graph.DirectedPseudograph;

import soot.*;
import soot.jimple.infoflow.android.SetupApplication;
import soot.jimple.toolkits.callgraph.CallGraph;
import soot.jimple.toolkits.callgraph.Edge;
import soot.options.Options;
import soot.tagkit.SourceLineNumberTag;
import soot.toolkits.graph.*;
import soot.toolkits.graph.pdg.MHGDominatorTree;
import soot.util.Chain;
import soot.util.queue.QueueReader;

public class APKCFG {

    public static void main(String[] args) throws Exception {
        APKCFG apkg = new APKCFG();

        // Sample application path and APK name for analysis.
        String appPath = "apks";
        String apk = "SampleApp"; // Renamed to a generic sample name.
        String apkPath = appPath + "/" + apk + ".apk";
        apkg.generateCFG(apkPath);
    }

    public void generateCFG(String appPath) throws Exception {
        // Set the path to the Android platform jar (update this path as needed)
        String androidPlatformPath = "path/to/android.jar";
        SetupApplication app = new SetupApplication(androidPlatformPath, appPath);
        Options.v().set_android_api_version(22);
        app.calculateSourcesSinksEntrypoints("SourcesAndSinks.txt");
        soot.G.reset();

        Options.v().set_src_prec(Options.src_prec_apk);
        Options.v().set_process_dir(Collections.singletonList(appPath));
        Options.v().set_force_android_jar(androidPlatformPath);
        Options.v().setPhaseOption("cg.spark", "on");
        Options.v().set_android_api_version(22);
        Options.v().set_whole_program(true);
        Options.v().set_allow_phantom_refs(true);
        Options.v().set_keep_line_number(true);
        Options.v().set_output_format(Options.output_format_jimple);
        app.setCallbackFile("AndroidCallbacks.txt");

        Scene.v().loadNecessaryClasses();

        SootMethod entryPoint = app.getEntryPointCreator().createDummyMain();
        Options.v().set_main_class(entryPoint.getSignature());
        Scene.v().setEntryPoints(Collections.singletonList(entryPoint));
        System.out.println(entryPoint.getActiveBody());

        PackManager.v().runPacks();

        printAugmentedCFG();
    }

    class CDNode {
        // Assume that Y is control dependent on X.
        private Unit nodeY;
        private int sourceLineNumberY;
        private Unit nodeX;
        private int sourceLineNumberX;
        int id;

        public CDNode(Unit nodeY, int sourceLineNumberY, Unit nodeX, int sourceLineNumberX, int id) {
            this.nodeY = nodeY;
            this.sourceLineNumberY = sourceLineNumberY;
            this.nodeX = nodeX;
            this.sourceLineNumberX = sourceLineNumberX;
            this.id = id;
        }
    }

    class CDEdge {
        private CDNode srcNode;
        private CDNode tgtNode;
        private int id;

        public CDEdge(CDNode srcNode, CDNode tgtNode, int id) {
            this.srcNode = srcNode;
            this.tgtNode = tgtNode;
            this.id = id;
        }
    }

    private void printAugmentedCFG() throws IOException {
        Chain<SootClass> applicationClasses = Scene.v().getApplicationClasses();
        for (SootClass sootClass : applicationClasses) {
            List<SootMethod> methods = sootClass.getMethods();
            for (SootMethod method : methods) {
                // Process only methods in MainActivity with the name "generateRandom".
                // (Alternatively, you could check for "onCreate" or any other method name.)
                if (!sootClass.getName().contains("MainActivity") || !method.getName().equals("generateRandom")) {
                    continue;
                }

                Body body = method.retrieveActiveBody();
                ExceptionalUnitGraph cfg = new ExceptionalUnitGraph(body);
                ExceptionalUnitGraph duplicateCfg = new ExceptionalUnitGraph(body);

                // Uncomment the following lines to export the CFG to DOT format if needed.
                // CFGToDotGraph cfgToDotGraph = new CFGToDotGraph();
                // DotGraph dotGraph = cfgToDotGraph.drawCFG(cfg, body);
                // dotGraph.plot("cfgs/" + method.getName() + ".out");

                List<Unit> entryNodeList = cfg.getHeads();
                if (entryNodeList.size() <= 0) {
                    System.out.println("*********************");
                    System.out.println("No CFG available");
                    System.out.println("*********************");
                    return;
                }
                Unit entryNode = entryNodeList.get(0);

                MHGPostDominatorsFinder<Unit> pdomFinder = new MHGPostDominatorsFinder(cfg);
                MHGDominatorTree<Unit> pdomTree = new MHGDominatorTree(pdomFinder);
                CytronDominanceFrontier<Unit> cdf = new CytronDominanceFrontier(pdomTree);

                int srcLineNo, srcLineNoOld = -1;

                // This map stores the source line number of each node,
                // because nodes in the dominator tree may lose their source line number.
                Map<Unit, Integer> nodeSourceLineNumberMap = new HashMap<>();
                nodeSourceLineNumberMapping(nodeSourceLineNumberMap, duplicateCfg, entryNodeList);

                Map<Unit, CDNode> cdNodeMap = new HashMap<>();
                DirectedPseudograph<CDNode, CDEdge> dg = new DirectedPseudograph<>(CDEdge.class);

                int nodeId = 0;
                int edgeId = 0;

                int entryNodeIndex = 1;
                int entryNodeSourceLineNumber = -1;

                for (Iterator<Unit> nodesIt = cfg.iterator(); nodesIt.hasNext(); ) {
                    Unit nodeY = nodesIt.next();
                    nodeId++;

                    if (entryNodeList.size() > 0 && entryNodeList.size() > entryNodeIndex && (entryNodeList.get(entryNodeIndex) + "").equals(nodeY + "")) {
                        entryNode = entryNodeList.get(entryNodeIndex);
                        srcLineNo = entryNode.getJavaSourceStartLineNumber();
                        srcLineNoOld = srcLineNo;
                        entryNodeSourceLineNumber = srcLineNo;
                        entryNodeIndex++;
                    }

                    srcLineNo = nodeY.getJavaSourceStartLineNumber();
                    if ((nodeY + "").equals("if $i0 <= $i1 goto $r0 = new java.util.Random")) {
                        int i = 0;
                    }

                    if (srcLineNo != entryNodeSourceLineNumber) {
                        // If two nodes are on the same source code line,
                        // the node may not have the source line number as an attribute.
                        srcLineNoOld = srcLineNo;
                    }

                    // Get the control dependency of the node.
                    List<DominatorNode<Unit>> domNodeList = cdf.getDominanceFrontierOf(pdomTree.getDode(nodeY));
                    Unit nodeX = null;
                    if (domNodeList.size() == 0) {
                        // The entry node can be more than one if the method throws exceptions.
                        nodeX = entryNode;
                    } else if (domNodeList.size() == 1) {
                        // This line of code is under a branch.
                        System.out.println("Branch condition encountered.");
                        nodeX = pdomTree.getDode(domNodeList.get(0).getGode()).getGode();

                        // If this line of code is also in a conditional statement,
                        // and its control dependency has the same source line number,
                        // then use the entry node instead.
                        if (nodeSourceLineNumberMap.get(nodeX) == srcLineNoOld) {
                            nodeX = entryNode;
                        }
                    } else if (domNodeList.size() >= 2) {
                        // This line of code depends on multiple branch conditions (e.g., if(a || b)).
                        Unit tmpNode = domNodeList.get(0).getGode();
                        for (int i = 1; i < domNodeList.size(); i++) {
                            if (nodeSourceLineNumberMap.get(tmpNode) != (nodeSourceLineNumberMap.get(domNodeList.get(i).getGode()))) {
                                // If a statement has more than two control dependency nodes,
                                // this situation is not clearly understood.
                            }
                        }
                        nodeX = domNodeList.get(0).getGode();
                    }

                    CDNode node = new CDNode(nodeY, srcLineNoOld, nodeX, nodeSourceLineNumberMap.get(nodeX), nodeId);
                    cdNodeMap.put(nodeY, node);
                    dg.addVertex(node);
                    System.out.println(nodeY + " --> Line " + srcLineNoOld + " || control dependent on " + nodeX + " --> Line " + nodeSourceLineNumberMap.get(nodeX));
                    System.out.println();
                }

                // Add edges to the graph.
                Queue<Unit> queue = new LinkedList<>(entryNodeList);
                // This set is used to avoid processing loops in the CFG.
                Set<Unit> addedNodeSet = new HashSet<>();

                while (!queue.isEmpty()) {
                    Unit srcNode = queue.poll();
                    List<Unit> successorNodeList = cfg.getSuccsOf(srcNode);
                    for (Iterator<Unit> successorNodeIt = successorNodeList.iterator(); successorNodeIt.hasNext(); ) {
                        edgeId++;
                        Unit succNode = successorNodeIt.next();
                        if (addedNodeSet.contains(succNode)) {
                            continue;
                        }
                        addedNodeSet.add(succNode);
                        queue.offer(succNode);
                        dg.addEdge(cdNodeMap.get(srcNode), cdNodeMap.get(succNode), new CDEdge(cdNodeMap.get(srcNode), cdNodeMap.get(succNode), edgeId));
                    }
                }

                DOTExporter<CDNode, CDEdge> exporter = new DOTExporter<>(new CDNodeIdProvider(), new CDNodeNameProvider(), null);
                Path path = new File("cfgs/" + sootClass.getName() + "/").toPath();
                if (!Files.exists(path)) {
                    Files.createDirectories(path);
                }
                exporter.exportGraph(dg, new FileWriter(path + "/" + method.getName() + ".out"));
            }
        }
        System.out.println();
    }

    private void nodeSourceLineNumberMapping(Map<Unit, Integer> nodeSourceLineNumberMap, ExceptionalUnitGraph cfg, List<Unit> entryNodeList) {
        int srcLineNo, srcLineNoOld = -1;

        for (Iterator<Unit> nodesIt = cfg.iterator(); nodesIt.hasNext(); ) {
            Unit node = nodesIt.next();

            srcLineNo = node.getJavaSourceStartLineNumber();
            if (srcLineNo != -1) {
                // If two nodes are on the same source code line,
                // the node may not have the source line number as an attribute.
                srcLineNoOld = srcLineNo;
            }
            nodeSourceLineNumberMap.put(node, srcLineNoOld);
        }
    }

    class CDNodeNameProvider implements ComponentNameProvider<CDNode> {
        @Override
        public String getName(CDNode cdNode) {
            String s = cdNode.nodeY + " --> Line " + cdNode.sourceLineNumberY
                    + " || control dependent on " + cdNode.nodeX + " --> Line " + cdNode.sourceLineNumberX;
            // Remove double quotes for DOT-to-SVG conversion.
            if (s.contains("\"")) {
                s = s.replace('\"', '\'');
            }
            return s;
        }
    }

    class CDEdgeLabelProvider implements ComponentNameProvider<CDEdge> {
        @Override
        public String getName(CDEdge cdEdge) {
            return cdEdge.toString();
        }
    }

    class CDNodeIdProvider implements ComponentNameProvider<CDNode> {
        @Override
        public String getName(CDNode cdNode) {
            return String.valueOf(cdNode.id);
        }
    }
}
